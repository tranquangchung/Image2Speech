import pdb

from tacotron2.load_model import load_model
from tacotron2.hparams import create_hparams, create_hparams_hubert100
from tacotron2.utils import load_vocoder, vocoder_infer
from scipy.io.wavfile import write
from tacotron2.tools import init_model, asr_english, init_model_vn, asr_vietnamese
import numpy as np
import torch
import os
from tacotron2.evaluation import evaluate_result, get_reference
import json

def evaluate_model(pathfolder, name_file_token, epoch, config):
    checkpoint_path = config['tacotron2_path']
    print("Load tacotron2 model from checkpoint: ")
    print(checkpoint_path)
    hparams = create_hparams()
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.cuda().eval()
    vocoder = load_vocoder()
    processor_asr, model_asr = init_model()

    print(pathfolder)
    print(name_file_token)
    path_caption = f"{pathfolder}/{name_file_token}.txt"
    path_folder_wave = f"{pathfolder}/{name_file_token}_tacotron2"
    os.makedirs(path_folder_wave, exist_ok=True)

    with open(path_caption, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    transcript_ars = {}

    with torch.no_grad():
        for i, text in enumerate(texts):
            name, text = text.split('|')
            name = name.split('.')[0]
            text = np.array([[int(x) for x in text.split()]])
            sequence = torch.autograd.Variable(
                torch.from_numpy(text)).cuda().long()
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence)
            audios = vocoder_infer(mel_outputs_postnet, vocoder)
            path2save = f"{path_folder_wave}/{name}.wav"
            write(path2save, 22050, audios[0])
            transcript = asr_english(path2save, model_asr, processor_asr)
            transcript_ars[name] = transcript[0].lower()
            print(f"{i}: {transcript[0].lower()}")
            # remove wave file
            os.remove(path2save)

    ### save transcript
    with open(f"{path_folder_wave}/transcript_ars.txt", 'w') as fp:
        for key, value in transcript_ars.items():
            fp.write(f"{key}|{value}\n")

    ### evaluate
    image_captions = get_reference(config['reference_path'])
    score = evaluate_result(image_captions, transcript_ars)
    # write score to file
    with open(f"{pathfolder}/score_{epoch}.txt", 'w') as fp:
        fp.write(str(score))
    print("*"*20)
    print(name_file_token)
    print(score)
    return score

def evaluate_model_vn(pathfolder, name_file_token, epoch, config):
    checkpoint_path = config['tacotron2_path']
    print("Load tacotron2 model from checkpoint: ")
    print(checkpoint_path)
    hparams = create_hparams()
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.cuda().eval()
    vocoder = load_vocoder()
    processor_asr, model_asr = init_model_vn()

    print(pathfolder)
    print(name_file_token)
    path_caption = f"{pathfolder}/{name_file_token}.txt"
    path_folder_wave = f"{pathfolder}/{name_file_token}_tacotron2"
    os.makedirs(path_folder_wave, exist_ok=True)

    with open(path_caption, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    transcript_ars = {}

    with torch.no_grad():
        for i, text in enumerate(texts):
            name, text = text.split('|')
            name = name.split('.')[0]
            text = np.array([[int(x) for x in text.split()]])
            sequence = torch.autograd.Variable(
                torch.from_numpy(text)).cuda().long()
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence)
            audios = vocoder_infer(mel_outputs_postnet, vocoder)
            path2save = f"{path_folder_wave}/{name}.wav"
            write(path2save, 22050, audios[0])
            transcript = asr_vietnamese(path2save, model_asr, processor_asr)
            transcript_ars[name] = transcript[0].lower()
            print(f"{i}: {transcript[0].lower()}")
            # remove wave file
            os.remove(path2save)

    ### save transcript
    with open(f"{path_folder_wave}/transcript_ars.txt", 'w') as fp:
        for key, value in transcript_ars.items():
            fp.write(f"{key}|{value}\n")

    ### evaluate
    image_captions = get_reference(config['reference_path'])
    score = evaluate_result(image_captions, transcript_ars)
    # write score to file
    with open(f"{pathfolder}/score_{epoch}.txt", 'w') as fp:
        fp.write(str(score))
    print("*"*20)
    print(name_file_token)
    print(score)
    return score

def evaluate_model_vn_hubert(pathfolder, name_file_token, epoch, config):
    checkpoint_path = config['tacotron2_path']
    print("Load tacotron2 model from checkpoint: ")
    print(checkpoint_path)
    hparams = create_hparams_hubert100()
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.cuda().eval()
    vocoder = load_vocoder()
    processor_asr, model_asr = init_model_vn()

    print(pathfolder)
    print(name_file_token)
    path_caption = f"{pathfolder}/{name_file_token}.txt"
    path_folder_wave = f"{pathfolder}/{name_file_token}_tacotron2"
    os.makedirs(path_folder_wave, exist_ok=True)

    with open(path_caption, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    transcript_ars = {}

    with torch.no_grad():
        for i, text in enumerate(texts):
            name, text = text.split('|')
            name = name.split('.')[0]
            text = np.array([[int(x) for x in text.split()]])
            sequence = torch.autograd.Variable(
                torch.from_numpy(text)).cuda().long()
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence)
            audios = vocoder_infer(mel_outputs_postnet, vocoder)
            path2save = f"{path_folder_wave}/{name}.wav"
            write(path2save, 22050, audios[0])
            transcript = asr_vietnamese(path2save, model_asr, processor_asr)
            transcript_ars[name] = transcript[0].lower()
            print(f"{i}: {transcript[0].lower()}")
            # remove wave file
            os.remove(path2save)

    ### save transcript
    with open(f"{path_folder_wave}/transcript_ars.txt", 'w') as fp:
        for key, value in transcript_ars.items():
            fp.write(f"{key}|{value}\n")

    ### evaluate
    image_captions = get_reference(config['reference_path'])
    score = evaluate_result(image_captions, transcript_ars)
    # write score to file
    with open(f"{pathfolder}/score_{epoch}.txt", 'w') as fp:
        fp.write(str(score))
    print("*"*20)
    print(name_file_token)
    print(score)
    return score