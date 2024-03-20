'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
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

from models.blip_unit import blip_decoder
from models.blip_unit import VIT_LLM
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate_model_vn


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{0}/{1}]'.format(epoch, config['max_epoch'])
    print_freq = 50

    for i, (image, labels, decoder_target, attention_mask, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        labels = labels.to(device)
        decoder_target = decoder_target.to(device)
        attention_mask = attention_mask.to(device)
        
        loss = model(image, labels, decoder_target, attention_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if i % print_freq == 0:
            logger.add_scalar('train/loss', loss.item(), epoch*len(data_loader)+i)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

@torch.no_grad()
def evaluate(model, data_loader, epoch, device, config):
    # evaluate
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Val Caption Epoch: [{0}/{1}]'.format(epoch, config['max_epoch'])
    print_freq = 50

    for i, (image, labels, decoder_target, attention_mask, _, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        labels = labels.to(device)
        decoder_target = decoder_target.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            loss = model(image, labels, decoder_target, attention_mask)
        metric_logger.update(loss=loss.item())

    logger.add_scalar('val/loss', metric_logger.meters["loss"].global_avg, epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return metric_logger.meters["loss"].global_avg

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
    train_dataset, val_dataset, test_dataset = create_dataset('flickr8k_train_vn', config)
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size'], config['batch_size'], 1],num_workers=[10,3,3],
                                                          is_trains=[True, False, False], collate_fns=[train_dataset.collate_fn,val_dataset.collate_fn,test_dataset.collate_fn_infer])

    #### Model #### 
    print("Creating model")
    path_url = config['path_url']
    model = VIT_LLM(med_config=os.path.join(config['med_config'], 'med_config.json'))
    model.load_state_dict(torch.load(path_url)['model'], strict=True)
    print("Loaded model from: ", path_url)
    # total parameters and trainable parameters
    # finetune text decoder
    # print(model)
    for name, param in model.named_parameters():
        if 'visual_encoder' in name:
            param.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    best = float('inf')
    best_epoch = 0
    tried_times = 0

    print("Start training")
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device, config)
        
        val_loss = evaluate(model, val_loader, epoch, device, config)
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }

        torch.save(save_obj, os.path.join(args.output_dir, f'{epoch}.pth'))
        # delete the previous checkpoint to save space
        prev_path = os.path.join(args.output_dir, f'{epoch-1}.pth')
        if os.path.exists(prev_path):
            os.remove(prev_path)

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            torch.save(save_obj, os.path.join(args.output_dir, 'best.pth'))
            tried_times = 0
        else:
            tried_times += 1
            if tried_times > config['early_stop']:
                break
                # pass # no early stop
        if epoch % 10 == 0 and epoch > 0:
            # load the best model
            path_txt = os.path.join(args.output_dir, f'inference_{epoch}.txt')
            inference(model, test_loader, device, path_txt)
            print("Start evaluation")
            evaluate_model_vn(args.output_dir, f'inference_{epoch}', f"{epoch}", config)

    ### inference
    print("Start inference")
    # load the best model
    path_txt = os.path.join(args.output_dir, f'inference_{best_epoch}.txt')
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best.pth'))['model'], strict=True)
    inference(model, test_loader, device, path_txt)
    print("Done inference")
    print("Start evaluation")
    evaluate_model_vn(args.output_dir, f'inference_{best_epoch}', f"{best_epoch}_best", config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    print(args)
    print("*"*50)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    print(config)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    log_path = os.path.join(args.output_dir, 'logs')
    logger = SummaryWriter(log_path)
    yaml.dump(config, open(os.path.join(args.output_dir, 'args.yaml'), 'w'))

    main(args, config)