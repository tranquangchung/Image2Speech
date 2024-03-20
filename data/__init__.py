import pdb

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode

from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from data.flick8k_dataset import flickr8k_train, flickr8k_train_auditory_feedback
from data.flick8k_dataset import flickr8k_train_auditory_feedback_attention, flickr8k_train_vn, flickr8k_train_vn_hubert
from data.flick8k_dataset import flickr8k_train_auditory_feedback_vn
from data.nocaps_dataset import nocaps_eval
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.vqa_dataset import vqa_dataset
from data.nlvr_dataset import nlvr_dataset
from data.pretrain_dataset import pretrain_dataset
from transform.randaugment import RandomAugment
from torchvision import transforms
# import torchvision.transforms as transforms


def create_dataset(dataset, config, min_scale=0.5):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    imsize_netg = 256
    # transform_train_netg = transforms.Compose([
    #     transforms.Resize((imsize_netg, imsize_netg)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    norm_netg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train_netg = transforms.Compose([
            transforms.Resize(int(imsize_netg * 76 / 64)),
            transforms.RandomCrop(imsize_netg),
            transforms.RandomHorizontalFlip(),
            norm_netg,
    ])

    if dataset=='flick8k_audio':
        train_dataset = flickr8k_train(transform_train, config['image_root'], config['path_unit'], split="train")
        val_dataset = flickr8k_train(transform_test, config['image_root'], config['path_unit'], split="val")
        test_dataset = flickr8k_train(transform_test, config['image_root'], config['path_unit'], split="test")
        return train_dataset, val_dataset, test_dataset

    elif dataset=='flick8k_audio_auditory_feedback':
        train_dataset = flickr8k_train_auditory_feedback(transform_train, config['image_root'], config['wavs_root'], config['path_unit'], split="train")
        val_dataset = flickr8k_train_auditory_feedback(transform_test, config['image_root'], config['wavs_root'], config['path_unit'], split="val")
        test_dataset = flickr8k_train_auditory_feedback(transform_test, config['image_root'], config['wavs_root'], config['path_unit'], split="test")
        return train_dataset, val_dataset, test_dataset

    elif dataset=='flick8k_audio_auditory_feedback_attention':
        train_dataset = flickr8k_train_auditory_feedback_attention(transform_train, config['image_root'], config['wavs_root'], config['path_unit'], split="train")
        val_dataset = flickr8k_train_auditory_feedback_attention(transform_test, config['image_root'], config['wavs_root'], config['path_unit'], split="val")
        test_dataset = flickr8k_train_auditory_feedback_attention(transform_test, config['image_root'], config['wavs_root'], config['path_unit'], split="test")
        return train_dataset, val_dataset, test_dataset

    elif dataset=='flickr8k_train_vn':
        train_dataset = flickr8k_train_vn(transform_train, config, split="train")
        val_dataset = flickr8k_train_vn(transform_test, config, split="val")
        test_dataset = flickr8k_train_vn(transform_test, config, split="test")
        return train_dataset, val_dataset, test_dataset

    elif dataset=='flick8k_audio_auditory_feedback_vn':
        train_dataset = flickr8k_train_auditory_feedback_vn(transform_train, config, split="train")
        val_dataset = flickr8k_train_auditory_feedback_vn(transform_test, config, split="val")
        test_dataset = flickr8k_train_auditory_feedback_vn(transform_test, config, split="test")
        return train_dataset, val_dataset, test_dataset

    elif dataset=='flickr8k_train_vn_hubert':
        train_dataset = flickr8k_train_vn_hubert(transform_train, config, split="train")
        val_dataset = flickr8k_train_vn_hubert(transform_test, config, split="val")
        test_dataset = flickr8k_train_vn_hubert(transform_test, config, split="test")
        return train_dataset, val_dataset, test_dataset

    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

