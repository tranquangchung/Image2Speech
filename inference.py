'''
Question about optimizing based on another model's output
https://discuss.pytorch.org/t/optimizing-based-on-another-models-output/6935/6
'''
import argparse
import os
import pdb

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils
from utils import cosine_lr_schedule, log_auditory_feedback_v3, step_lr_schedule, cosine_step_schedule, \
    cosine_step_schedule_v2
from data import create_dataset, create_sampler, create_loader
from models.blip_unit_auditory_feedback import VIT_LLM_TTS_DecoderV4, VIT_LLM_TTS_DecoderV5, VIT_LLM_TTS_DecoderV6
from tacotron2.utils import load_vocoder, vocoder_infer
from scipy.io.wavfile import write
from evaluation import evaluate_model


def inference(model, data_loader, device, pathtxt):
    # evaluate
    model.eval()
    fout = open(pathtxt, 'w')
    for i, (image, image_ids) in enumerate(data_loader):
        image = image.to(device)
        with torch.no_grad():
            tokens = model.generate(image, sample=False, top_p=0.9, max_length=100, min_length=30, num_beams=5)
            if tokens[-1] != 1025:
                tokens = tokens + [1025]
            str_tokens = " ".join([str(x) for x in tokens])
            print('str_tokens: ', str_tokens)
            fout.write(image_ids[0] + "|" + str_tokens + '\n')
            if i % 100 == 0:
                fout.flush()
        # if i == 10: break

def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating captioning dataset")
    # train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)
    train_dataset, val_dataset, test_dataset = create_dataset('flick8k_audio_auditory_feedback_attention', config)
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[1, 1, 1],
                                                          num_workers=[10, 3, 3],
                                                          is_trains=[False, False, False],
                                                          collate_fns=[train_dataset.collate_fn,
                                                                       val_dataset.collate_fn_infer,
                                                                       test_dataset.collate_fn_infer])

    #### Model ####
    print("Creating model")
    path_folder = "/home/ldap-users/s2220411/Code/new_explore_tts/BLIP/output/auditory_feedback_v10/auditory_feedback_ftlm_fs_4layer_cosine_MSEhidden_feature"
    epoch = "46"
    path_url = os.path.join(path_folder, f'{epoch}.pth')
    model = VIT_LLM_TTS_DecoderV5(med_config=os.path.join(config['med_config'], 'med_config.json'))
    # model = VIT_LLM_TTS_DecoderV6(med_config=os.path.join(config['med_config'], 'med_config.json'))
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    model.load_state_dict(torch.load(path_url)['model'], strict=True)
    model = model.to(device)
    print("Loading at epoch: ", torch.load(path_url)['epoch'])
    print("Loading from: ", path_url)

    print("Start inference")
    # load the best model
    path_txt = os.path.join(path_folder, f'test_inference_{epoch}.txt')
    inference(model, test_loader, device, path_txt)
    print("Done inference")

    print("Start evaluation")
    evaluate_model(path_folder, f'test_inference_{epoch}', f"{epoch}_best", config)
    print("path_url: ", path_url)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    print(args)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args, config)