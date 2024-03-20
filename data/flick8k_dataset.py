import os
import json
import pdb

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import torch
from PIL import Image
import numpy as np
from data.utils import pre_caption
from tacotron2 import layers
from tacotron2.utils import load_wav_to_torch

def remove_consecutive_duplicates(sequence):
    numbers = sequence.split()
    distinct_numbers = [numbers[i] for i in range(len(numbers)) if i == 0 or numbers[i] != numbers[i - 1]]
    return ' '.join(distinct_numbers)

class flickr8k_train(Dataset):
    def __init__(self, transform, image_root, path_unit, split, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.split = split
        self.load_phonemes(path_unit, split)
        self.start = 1024
        self.end = 1025

    def load_phonemes(self, path_unit, split):
        filepath = '%s/%s.txt' % (path_unit, split)
        self.phonemes_dict = {}
        self.image_ids = []
        if self.split == 'train' or self.split == 'val':
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    phonemes = f.readlines()
                    # process phonemes
                    for phoneme in phonemes:
                        image_id, phone_id = phoneme.strip().split('|')
                        image_id = image_id.split('/')[1].split('.')[0]
                        self.phonemes_dict[image_id] = remove_consecutive_duplicates(phone_id)
                        self.image_ids.append(image_id)
            print('Load %d phonemes from %s' % (len(self.phonemes_dict), filepath))
            print('Load %d images from %s' % (len(self.image_ids), filepath))
        else:
            with open(filepath, 'r') as f:
                phonemes = f.readlines()
                # process phonemes
                for phoneme in phonemes:
                    image_id, phone_id = phoneme.strip().split('|')
                    image_id = image_id.split('/')[1].split('.')[0][:-2]
                    self.image_ids.append(image_id)
                self.image_ids = list(set(self.image_ids))
            print('Load %d images from %s' % (len(self.image_ids), filepath))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'val':
            item_id = self.image_ids[index]
            name_image = item_id[:-2]
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image_raw = Image.open(image_path).convert('RGB')
            image = self.transform(image_raw)
            phone = self.phonemes_dict[item_id]
            phone = [self.start] + [int(x) for x in phone.split()] + [self.end]
            phone_len = len(phone)
            return image, phone, phone_len, item_id
        else:
            item_id = self.image_ids[index]
            name_image = item_id
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, item_id

    def collate_fn(self, batch):
        # sorted by phone_len
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        images, phones, phone_len, item_ids = zip(*batch)
        max_phone_len = max(phone_len)
        images = torch.stack(images, 0)
        # padding for phones
        labels = []
        decoder_target = []
        attention_mask = []
        for phone in phones:
            phone_tmp = phone + [0] * (max_phone_len - len(phone))
            labels.append(phone_tmp)
            phone_tmp1 = phone + [-100] * (max_phone_len - len(phone))
            decoder_target.append(phone_tmp1)
            phone_tmp2 = [1] * len(phone) + [0] * (max_phone_len - len(phone))
            attention_mask.append(phone_tmp2)
        labels = torch.LongTensor(labels)
        decoder_target = torch.LongTensor(decoder_target)
        attention_mask = torch.LongTensor(attention_mask)
        phone_len = torch.LongTensor(phone_len)
        return images, labels, decoder_target, attention_mask, phone_len, item_ids

    def collate_fn_infer(self, batch):
        images, item_ids = zip(*batch)
        images = torch.stack(images, 0)
        return images, item_ids


class flickr8k_train_auditory_feedback(Dataset):
    def __init__(self, transform, image_root, wavs_root, path_unit, split, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.transform = transform
        self.image_root = image_root
        self.wavs_root = wavs_root
        self.max_words = max_words
        self.prompt = prompt
        self.split = split
        self.load_phonemes(path_unit, split)
        self.start = 1024
        self.end = 1025
        self.stft = layers.TacotronSTFT(1024, 256, 1024, 80, 22050, 0.0, 8000.0)
        self.n_frames_per_step = 1
        self.max_wav_value = 32768.0
        self.attention_root = "/home/ldap-users/s2220411/Code/new_explore_tts/Show-and-Speak/Data_for_SAS/tacotron2_attention/vq3_flick8k"

    def load_phonemes(self, path_unit, split):
        filepath = '%s/%s.txt' % (path_unit, split)
        self.phonemes_dict = {}
        self.image_ids = []
        if self.split == 'train' or self.split == 'val':
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    phonemes = f.readlines()
                    # process phonemes
                    for phoneme in phonemes:
                        image_id, phone_id = phoneme.strip().split('|')
                        image_id = image_id.split('/')[1].split('.')[0]
                        self.phonemes_dict[image_id] = remove_consecutive_duplicates(phone_id)
                        self.image_ids.append(image_id)
            print('Load %d phonemes from %s' % (len(self.phonemes_dict), filepath))
            print('Load %d images from %s' % (len(self.image_ids), filepath))
        else:
            with open(filepath, 'r') as f:
                phonemes = f.readlines()
                # process phonemes
                for phoneme in phonemes:
                    image_id, phone_id = phoneme.strip().split('|')
                    image_id = image_id.split('/')[1].split('.')[0][:-2]
                    self.image_ids.append(image_id)
                self.image_ids = list(set(self.image_ids))
            print('Load %d images from %s' % (len(self.image_ids), filepath))

    def __len__(self):
        return len(self.image_ids)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / audio.abs().max()
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'val':
            item_id = self.image_ids[index]
            name_image = item_id[:-2]
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image_raw = Image.open(image_path).convert('RGB')
            image = self.transform(image_raw)
            phone = self.phonemes_dict[item_id]
            phone = [self.start] + [int(x) for x in phone.split()] + [self.end]
            phone_len = len(phone)
            # load mel-spectrogram
            mel_path = os.path.join(self.wavs_root, item_id + '.wav')
            mel = self.get_mel(mel_path)
            # load Pre-Alignment Guided Attention
            # attention_path = os.path.join(self.attention_root, item_id + '.npy')
            # attention = np.load(attention_path)
            return image, phone, phone_len, mel, item_id
        else:
            item_id = self.image_ids[index]
            name_image = item_id
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, item_id

    def collate_fn(self, batch):
        # sorted by phone_len
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        images, phones, phone_len, mels, item_ids = zip(*batch)
        max_phone_len = max(phone_len)
        images = torch.stack(images, 0)
        # padding for phones
        labels = []
        decoder_target = []
        attention_mask = []
        for phone in phones:
            phone_tmp = phone + [0] * (max_phone_len - len(phone))
            labels.append(phone_tmp)
            phone_tmp1 = phone + [-100] * (max_phone_len - len(phone))
            decoder_target.append(phone_tmp1)
            phone_tmp2 = [1] * len(phone) + [0] * (max_phone_len - len(phone))
            attention_mask.append(phone_tmp2)
        labels = torch.LongTensor(labels)
        decoder_target = torch.LongTensor(decoder_target)
        attention_mask = torch.LongTensor(attention_mask)
        phone_len = torch.LongTensor(phone_len)
        ################################
        # Right zero-pad mel-spec
        num_mels = mels[0].size(0)
        max_target_len = max([x.size(1) for x in mels])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(mels), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(mels), max_target_len)
        gate_padded.zero_()
        mel_lengths = torch.LongTensor(len(batch))
        for i in range(len(mels)):
            mel = mels[i]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            mel_lengths[i] = mel.size(1)

        return (images, labels, decoder_target, attention_mask, phone_len,
                mel_padded, gate_padded, mel_lengths, item_ids)

    def collate_fn_infer(self, batch):
        images, item_ids = zip(*batch)
        images = torch.stack(images, 0)
        return images, item_ids

class flickr8k_train_auditory_feedback_attention(Dataset):
    def __init__(self, transform, image_root, wavs_root, path_unit, split, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.transform = transform
        self.image_root = image_root
        self.wavs_root = wavs_root
        self.max_words = max_words
        self.prompt = prompt
        self.split = split
        self.load_phonemes(path_unit, split)
        self.start = 1024
        self.end = 1025
        self.stft = layers.TacotronSTFT(1024, 256, 1024, 80, 22050, 0.0, 8000.0)
        self.n_frames_per_step = 1
        self.max_wav_value = 32768.0

    def load_phonemes(self, path_unit, split):
        filepath = '%s/%s.txt' % (path_unit, split)
        self.phonemes_dict = {}
        self.image_ids = []
        if self.split == 'train' or self.split == 'val':
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    phonemes = f.readlines()
                    # process phonemes
                    for phoneme in phonemes:
                        image_id, phone_id = phoneme.strip().split('|')
                        image_id = image_id.split('/')[1].split('.')[0]
                        self.phonemes_dict[image_id] = remove_consecutive_duplicates(phone_id)
                        self.image_ids.append(image_id)
            print('Load %d phonemes from %s' % (len(self.phonemes_dict), filepath))
            print('Load %d images from %s' % (len(self.image_ids), filepath))
        else:
            with open(filepath, 'r') as f:
                phonemes = f.readlines()
                # process phonemes
                for phoneme in phonemes:
                    image_id, phone_id = phoneme.strip().split('|')
                    image_id = image_id.split('/')[1].split('.')[0][:-2]
                    self.image_ids.append(image_id)
                self.image_ids = list(set(self.image_ids))
            print('Load %d images from %s' % (len(self.image_ids), filepath))

    def __len__(self):
        return len(self.image_ids)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / audio.abs().max()
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'val':
            item_id = self.image_ids[index]
            name_image = item_id[:-2]
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image_raw = Image.open(image_path).convert('RGB')
            image = self.transform(image_raw)
            phone = self.phonemes_dict[item_id]
            phone = [self.start] + [int(x) for x in phone.split()] + [self.end]
            phone_len = len(phone)
            # load mel-spectrogram
            mel_path = os.path.join(self.wavs_root, item_id + '.wav')
            mel = self.get_mel(mel_path)
            return image, phone, phone_len, mel, item_id
        else:
            item_id = self.image_ids[index]
            name_image = item_id
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, item_id

    def collate_fn(self, batch):
        # sorted by phone_len
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        images, phones, phone_len, mels, item_ids = zip(*batch)
        max_phone_len = max(phone_len)
        images = torch.stack(images, 0)
        # padding for phones
        labels = []
        decoder_target = []
        attention_mask = []
        for phone in phones:
            phone_tmp = phone + [0] * (max_phone_len - len(phone))
            labels.append(phone_tmp)
            phone_tmp1 = phone + [-100] * (max_phone_len - len(phone))
            decoder_target.append(phone_tmp1)
            phone_tmp2 = [1] * len(phone) + [0] * (max_phone_len - len(phone))
            attention_mask.append(phone_tmp2)
        labels = torch.LongTensor(labels)
        decoder_target = torch.LongTensor(decoder_target)
        attention_mask = torch.LongTensor(attention_mask)
        phone_len = torch.LongTensor(phone_len)
        ################################
        # Right zero-pad mel-spec
        num_mels = mels[0].size(0)
        max_target_len = max([x.size(1) for x in mels])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded and attention
        mel_padded = torch.FloatTensor(len(mels), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(mels), max_target_len)
        gate_padded.zero_()
        mel_lengths = torch.LongTensor(len(batch))
        for i in range(len(mels)):
            mel = mels[i]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            mel_lengths[i] = mel.size(1)
        return (images, labels, decoder_target, attention_mask, phone_len,
                mel_padded, gate_padded, mel_lengths, item_ids)

    def collate_fn_infer(self, batch):
        images, item_ids = zip(*batch)
        images = torch.stack(images, 0)
        return images, item_ids

class flickr8k_train_vn(Dataset):
    def __init__(self, transform, configs, split, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.transform = transform
        self.image_root = configs['image_root']
        self.max_words = max_words
        self.prompt = prompt
        self.split = split
        self.percent = configs["percent"]
        self.load_phonemes(configs["path_unit"], split)
        self.start = 1024
        self.end = 1025

    def load_phonemes(self, path_unit, split):
        filepath = '%s/%s.txt' % (path_unit, split)
        self.phonemes_dict = {}
        self.image_ids = []
        if self.split == 'train' or self.split == 'val':
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    phonemes = f.readlines()
                    # process phonemes
                    for phoneme in phonemes:
                        image_id, phone_id = phoneme.strip().split('|')
                        image_id = image_id.split('/')[1].split('.')[0]
                        self.phonemes_dict[image_id] = remove_consecutive_duplicates(phone_id)
                        self.image_ids.append(image_id)
            if self.percent < 1 and self.split == 'train':
                self.image_ids = self.image_ids[:int(len(self.image_ids) * self.percent)]
            print('Load %d phonemes from %s' % (len(self.phonemes_dict), filepath)) # we keep the same number of phonemes because it doesn't affect to ...
            print('Load %d images from %s' % (len(self.image_ids), filepath))
        else:
            with open(filepath, 'r') as f:
                phonemes = f.readlines()
                # process phonemes
                for phoneme in phonemes:
                    image_id, phone_id = phoneme.strip().split('|')
                    image_id = image_id.split('/')[1].split('.')[0][:-2]
                    self.image_ids.append(image_id)
                self.image_ids = list(set(self.image_ids))
            print('Load %d images from %s' % (len(self.image_ids), filepath))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'val':
            item_id = self.image_ids[index]
            name_image = item_id[:-2]
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image_raw = Image.open(image_path).convert('RGB')
            image = self.transform(image_raw)
            phone = self.phonemes_dict[item_id]
            phone = [self.start] + [int(x) for x in phone.split()] + [self.end]
            phone_len = len(phone)
            return image, phone, phone_len, item_id
        else:
            item_id = self.image_ids[index]
            name_image = item_id
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, item_id

    def collate_fn(self, batch):
        # sorted by phone_len
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        images, phones, phone_len, item_ids = zip(*batch)
        max_phone_len = max(phone_len)
        images = torch.stack(images, 0)
        # padding for phones
        labels = []
        decoder_target = []
        attention_mask = []
        for phone in phones:
            phone_tmp = phone + [0] * (max_phone_len - len(phone))
            labels.append(phone_tmp)
            phone_tmp1 = phone + [-100] * (max_phone_len - len(phone))
            decoder_target.append(phone_tmp1)
            phone_tmp2 = [1] * len(phone) + [0] * (max_phone_len - len(phone))
            attention_mask.append(phone_tmp2)
        labels = torch.LongTensor(labels)
        decoder_target = torch.LongTensor(decoder_target)
        attention_mask = torch.LongTensor(attention_mask)
        phone_len = torch.LongTensor(phone_len)
        return images, labels, decoder_target, attention_mask, phone_len, item_ids

    def collate_fn_infer(self, batch):
        images, item_ids = zip(*batch)
        images = torch.stack(images, 0)
        return images, item_ids

class flickr8k_train_auditory_feedback_vn(Dataset):
    def __init__(self, transform, configs, split, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.transform = transform
        self.image_root = configs["image_root"]
        self.wavs_root = configs["wavs_root"]
        self.max_words = max_words
        self.prompt = prompt
        self.split = split
        self.percent = configs["percent"]
        self.load_phonemes(configs["path_unit"], split)
        self.start = 1024
        self.end = 1025
        self.stft = layers.TacotronSTFT(1024, 256, 1024, 80, 22050, 0.0, 8000.0)
        self.n_frames_per_step = 1
        self.max_wav_value = 32768.0

    def load_phonemes(self, path_unit, split):
        filepath = '%s/%s.txt' % (path_unit, split)
        self.phonemes_dict = {}
        self.image_ids = []
        if self.split == 'train' or self.split == 'val':
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    phonemes = f.readlines()
                    # process phonemes
                    for phoneme in phonemes:
                        image_id, phone_id = phoneme.strip().split('|')
                        image_id = image_id.split('/')[1].split('.')[0]
                        self.phonemes_dict[image_id] = remove_consecutive_duplicates(phone_id)
                        self.image_ids.append(image_id)
            if self.percent < 1 and self.split == 'train':
                self.image_ids = self.image_ids[:int(len(self.image_ids) * self.percent)]
            print('Load %d phonemes from %s' % (len(self.phonemes_dict), filepath))
            print('Load %d images from %s' % (len(self.image_ids), filepath))
        else:
            with open(filepath, 'r') as f:
                phonemes = f.readlines()
                # process phonemes
                for phoneme in phonemes:
                    image_id, phone_id = phoneme.strip().split('|')
                    image_id = image_id.split('/')[1].split('.')[0][:-2]
                    self.image_ids.append(image_id)
                self.image_ids = list(set(self.image_ids))
            print('Load %d images from %s' % (len(self.image_ids), filepath))

    def __len__(self):
        return len(self.image_ids)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / audio.abs().max()
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'val':
            item_id = self.image_ids[index]
            name_image = item_id[:-2]
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image_raw = Image.open(image_path).convert('RGB')
            image = self.transform(image_raw)
            phone = self.phonemes_dict[item_id]
            phone = [self.start] + [int(x) for x in phone.split()] + [self.end]
            phone_len = len(phone)
            # load mel-spectrogram
            mel_path = os.path.join(self.wavs_root, item_id + '.wav')
            mel = self.get_mel(mel_path)
            return image, phone, phone_len, mel, item_id
        else:
            item_id = self.image_ids[index]
            name_image = item_id
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, item_id

    def collate_fn(self, batch):
        # sorted by phone_len
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        images, phones, phone_len, mels, item_ids = zip(*batch)
        max_phone_len = max(phone_len)
        images = torch.stack(images, 0)
        # padding for phones
        labels = []
        decoder_target = []
        attention_mask = []
        for phone in phones:
            phone_tmp = phone + [0] * (max_phone_len - len(phone))
            labels.append(phone_tmp)
            phone_tmp1 = phone + [-100] * (max_phone_len - len(phone))
            decoder_target.append(phone_tmp1)
            phone_tmp2 = [1] * len(phone) + [0] * (max_phone_len - len(phone))
            attention_mask.append(phone_tmp2)
        labels = torch.LongTensor(labels)
        decoder_target = torch.LongTensor(decoder_target)
        attention_mask = torch.LongTensor(attention_mask)
        phone_len = torch.LongTensor(phone_len)
        ################################
        # Right zero-pad mel-spec
        num_mels = mels[0].size(0)
        max_target_len = max([x.size(1) for x in mels])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(mels), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(mels), max_target_len)
        gate_padded.zero_()
        mel_lengths = torch.LongTensor(len(batch))
        for i in range(len(mels)):
            mel = mels[i]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            mel_lengths[i] = mel.size(1)
        return (images, labels, decoder_target, attention_mask, phone_len,
                mel_padded, gate_padded, mel_lengths, item_ids)

    def collate_fn_infer(self, batch):
        images, item_ids = zip(*batch)
        images = torch.stack(images, 0)
        return images, item_ids

class flickr8k_train_vn_hubert(Dataset):
    def __init__(self, transform, configs, split, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.transform = transform
        self.image_root = configs['image_root']
        self.max_words = max_words
        self.prompt = prompt
        self.split = split
        self.percent = configs["percent"]
        self.load_phonemes(configs["path_unit"], split)
        self.start = 100
        self.end = 101

    def load_phonemes(self, path_unit, split):
        filepath = '%s/%s.txt' % (path_unit, split)
        self.phonemes_dict = {}
        self.image_ids = []
        if self.split == 'train' or self.split == 'val':
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    phonemes = f.readlines()
                    # process phonemes
                    for phoneme in phonemes:
                        image_id, phone_id = phoneme.strip().split('|')
                        image_id = image_id.split('/')[1].split('.')[0]
                        self.phonemes_dict[image_id] = remove_consecutive_duplicates(phone_id)
                        self.image_ids.append(image_id)
            if self.percent < 1 and self.split == 'train':
                self.image_ids = self.image_ids[:int(len(self.image_ids) * self.percent)]
            print('Load %d phonemes from %s' % (len(self.phonemes_dict), filepath)) # we keep the same number of phonemes because it doesn't affect to ...
            print('Load %d images from %s' % (len(self.image_ids), filepath))
        else:
            with open(filepath, 'r') as f:
                phonemes = f.readlines()
                # process phonemes
                for phoneme in phonemes:
                    image_id, phone_id = phoneme.strip().split('|')
                    image_id = image_id.split('/')[1].split('.')[0][:-2]
                    self.image_ids.append(image_id)
                self.image_ids = list(set(self.image_ids))
            print('Load %d images from %s' % (len(self.image_ids), filepath))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'val':
            item_id = self.image_ids[index]
            name_image = item_id[:-2]
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image_raw = Image.open(image_path).convert('RGB')
            image = self.transform(image_raw)
            phone = self.phonemes_dict[item_id]
            phone = [self.start] + [int(x) for x in phone.split()] + [self.end]
            phone_len = len(phone)
            return image, phone, phone_len, item_id
        else:
            item_id = self.image_ids[index]
            name_image = item_id
            image_path = os.path.join(self.image_root, name_image + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, item_id

    def collate_fn(self, batch):
        # sorted by phone_len
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        images, phones, phone_len, item_ids = zip(*batch)
        max_phone_len = max(phone_len)
        images = torch.stack(images, 0)
        # padding for phones
        labels = []
        decoder_target = []
        attention_mask = []
        for phone in phones:
            phone_tmp = phone + [0] * (max_phone_len - len(phone))
            labels.append(phone_tmp)
            phone_tmp1 = phone + [-100] * (max_phone_len - len(phone))
            decoder_target.append(phone_tmp1)
            phone_tmp2 = [1] * len(phone) + [0] * (max_phone_len - len(phone))
            attention_mask.append(phone_tmp2)
        labels = torch.LongTensor(labels)
        decoder_target = torch.LongTensor(decoder_target)
        attention_mask = torch.LongTensor(attention_mask)
        phone_len = torch.LongTensor(phone_len)
        return images, labels, decoder_target, attention_mask, phone_len, item_ids

    def collate_fn_infer(self, batch):
        images, item_ids = zip(*batch)
        images = torch.stack(images, 0)
        return images, item_ids