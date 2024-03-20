'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import pdb
import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_vit import ViT

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from models.GAN import NetG
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.DAMSM import RNN_ENCODER
from scipy.stats import truncnorm
import numpy as np
from torch.nn import CrossEntropyLoss
from tacotron2.model import Decoder as TTS_Decoder
from tacotron2.model import Postnet, Encoder as TTS_Encoder
from Transformer_TTS.network import MelDecoder
from tacotron2.hparams import create_hparams
from collections import OrderedDict

class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)

        
    def forward(self, image, labels, decoder_target, attention_mask):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        decoder_output = self.text_decoder(labels,
                                           attention_mask = attention_mask,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_target,
                                           return_dict = True,   
                                          )
        # loss_lm = decoder_output.loss
        # return loss_lm
        return decoder_output
        
    def generate(self, image, sample=False, num_beams=5, max_length=40, min_length=30, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}

        input_ids = torch.tensor([1024]).unsqueeze(0).to(image.device)
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=1025,
                                                  pad_token_id=0,
                                                  repetition_penalty=1.1,
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=1025,
                                                  pad_token_id=0,
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)
        return outputs[0].cpu().numpy().tolist()
    

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        # assert(len(msg.missing_keys)==0)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    ignore_keys = ["text_encoder.cls.decoder"]
    for key in model.state_dict().keys():
        if key in state_dict.keys() and key not in ignore_keys:
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=True)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class VIT_LLM(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        vit_model = ViT('B_16_imagenet1k', pretrained=True)
        del vit_model.fc
        del vit_model.norm
        self.visual_encoder = vit_model
        med_config = BertConfig.from_json_file(med_config)
        vision_width = 768
        med_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med_config)
        self.text_decoder.apply(weights_init_xavier)

    def forward(self, image, labels, decoder_target, attention_mask):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        # loss_lm = decoder_output.loss
        # return loss_lm
        return decoder_output

    def generate(self, image, sample=False, num_beams=5, max_length=40, min_length=30, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = torch.tensor([1024]).unsqueeze(0).to(image.device)
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        return outputs[0].cpu().numpy().tolist()

def process_predicted_units(predicted_units, length_units):
    max_phone_len = max(length_units)
    padding_predicted_units = []
    for phone in predicted_units:
        phone_tmp = phone + [0] * (max_phone_len - len(phone))
        padding_predicted_units.append(phone_tmp)
    padding_predicted_units = torch.LongTensor(padding_predicted_units)
    padding_predicted_units = padding_predicted_units.to('cuda')
    length_units = torch.LongTensor(length_units).to('cuda')
    return padding_predicted_units, length_units

def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    values = truncation * values
    noise = torch.tensor(values, dtype=torch.float).to('cuda')
    return noise

def encode_tokens(text_encoder, units, unit_lens):
    # encode text
    # units = torch.tensor(units).cuda()
    # unit_lens = torch.tensor(unit_lens).cuda()
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(units.size(0))
        else:
            hidden = text_encoder.init_hidden(units.size(0))
        words_embs, sent_emb = text_encoder(units, unit_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs

def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap

def sort_sents(captions, caption_lens):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()
    captions = torch.tensor(captions).cuda()
    sorted_cap_lens = torch.tensor(sorted_cap_lens).cuda()
    return captions, sorted_cap_lens, sorted_cap_indices

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Embedding_Encoder(nn.Module):
    '''Encoder module:
        - two layer FC
    '''
    def __init__(self,in_dim, encoder_embedding_dim):
        super(Embedding_Encoder, self).__init__()
        in_sizes = [in_dim] + [encoder_embedding_dim*2]
        sizes = [encoder_embedding_dim*2] + [encoder_embedding_dim]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.2, training=True)
        return x

class VIT_LLM_TTS_Decoder(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        vit_model = ViT('B_16_imagenet1k', pretrained=True)
        del vit_model.fc
        del vit_model.norm
        self.visual_encoder = vit_model
        med_config = BertConfig.from_json_file(med_config)
        vision_width = 768
        med_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med_config)
        self.text_decoder.apply(weights_init_xavier)
        hparams = create_hparams()
        self.tts_decoder = TTS_Decoder(hparams)
        ### Load pretrained tacotron2 model
        self.fc = nn.Linear(med_config.vocab_size, 512)
        # self.fc = Embedding_Encoder(med_config.vocab_size, 512)

    def forward(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        text_embedding = decoder_output.logits
        text_embedding = self.fc(text_embedding)
        mel_outputs, gate_outputs, alignments = self.tts_decoder(text_embedding, mels, memory_lengths=text_lengths)

        return decoder_output, mel_outputs, gate_outputs, alignments

    def generate(self, image, sample=False, num_beams=5, max_length=40, min_length=30, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = torch.tensor([1024]).unsqueeze(0).to(image.device)
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        return outputs[0].cpu().numpy().tolist()

class VIT_LLM_TTS_DecoderV2(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        vit_model = ViT('B_16_imagenet1k', pretrained=True)
        del vit_model.fc
        del vit_model.norm
        self.visual_encoder = vit_model
        med_config = BertConfig.from_json_file(med_config)
        vision_width = 768
        med_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med_config)
        hparams = create_hparams()
        # self.tts_decoder = TTS_Decoder(hparams)
        ### Load pretrained tacotron2 model
        self.fc = nn.Linear(med_config.vocab_size, 512)
        self.tts_decoder = MelDecoder(512)
        self.init_weights()

        # self.fc = Embedding_Encoder(med_config.vocab_size, 512)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.text_decoder.weight, -initrange, initrange)
        nn.init.uniform_(self.tts_decoder.weight, -initrange, initrange)

    def forward(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        text_embedding = decoder_output.logits
        text_embedding = self.fc(text_embedding)
        # process for transformer tts
        pos_texts = []
        pos_mels = []
        for i in range(len(mel_lengths)):
            pos_text = np.arange(1, text_lengths[i].item() + 1)
            pos_mel = np.arange(1, mel_lengths[i].item() + 1)
            pos_texts.append(pos_text)
            pos_mels.append(pos_mel)
        pos_mels = _prepare_data(pos_mels).astype(np.int32)
        pos_texts = _prepare_data(pos_texts).astype(np.int32)
        pos_mels = torch.LongTensor(pos_mels).to('cuda')
        pos_texts = torch.LongTensor(pos_texts).to('cuda')
        c_mask = pos_texts.ne(0).type(torch.float)
        mels = mels.permute(0, 2, 1)
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.tts_decoder.forward(text_embedding, mels, c_mask, pos=pos_mels)
        mel_outputs = mel_output.permute(0, 2, 1)
        postnet_output = postnet_output.permute(0, 2, 1)
        return decoder_output, mel_outputs, postnet_output, attns_dec

    def generate(self, image, sample=False, num_beams=5, max_length=40, min_length=30, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = torch.tensor([1024]).unsqueeze(0).to(image.device)
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        return outputs[0].cpu().numpy().tolist()

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

class VIT_LLM_TTS_DecoderV3(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        vit_model = ViT('B_16_imagenet1k', pretrained=True)
        del vit_model.fc
        del vit_model.norm
        self.visual_encoder = vit_model
        med_config = BertConfig.from_json_file(med_config)
        vision_width = 768
        med_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med_config)
        self.text_decoder.apply(weights_init_xavier)
        hparams = create_hparams()
        self.tts_decoder = TTS_Decoder(hparams)
        self.postnet = Postnet(hparams)
        ### Load pretrained tacotron2 model
        self.fc = nn.Linear(med_config.vocab_size, 512)

    def forward(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        text_embedding = decoder_output.logits
        text_embedding = self.fc(text_embedding)
        mel_outputs, gate_outputs, alignments = self.tts_decoder(text_embedding, mels, memory_lengths=text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return decoder_output, mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def generate(self, image, sample=False, num_beams=5, max_length=40, min_length=30, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = torch.tensor([1024]).unsqueeze(0).to(image.device)
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        return outputs[0].cpu().numpy().tolist()


    def synthesis(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        text_embedding = decoder_output.logits
        text_embedding = self.fc(text_embedding)
        mel_outputs, gate_outputs, alignments = self.tts_decoder.inference(text_embedding)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return mel_outputs, mel_outputs_postnet

class VIT_LLM_TTS_DecoderV4(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        vit_model = ViT('B_16_imagenet1k', pretrained=True)
        del vit_model.fc
        del vit_model.norm
        self.visual_encoder = vit_model
        med_config = BertConfig.from_json_file(med_config)
        vision_width = 768
        med_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med_config)
        self.text_decoder.apply(weights_init_xavier)
        hparams = create_hparams()
        self.tts_decoder = TTS_Decoder(hparams)
        self.postnet = Postnet(hparams)
        ### Load pretrained tacotron2 model
        # self.fc = nn.Linear(med_config.vocab_size, 512)
        self.fc = nn.Linear(1024, 512)

    def forward(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        # text_embedding = decoder_output.logits
        text_embedding = decoder_output.hidden_states
        text_embedding = self.fc(text_embedding)
        mel_outputs, gate_outputs, alignments = self.tts_decoder(text_embedding, mels, memory_lengths=text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return decoder_output, mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def generate(self, image, sample=False, num_beams=5, max_length=40, min_length=30, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = torch.tensor([1024]).unsqueeze(0).to(image.device)
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        return outputs[0].cpu().numpy().tolist()


    def synthesis(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        # text_embedding = decoder_output.logits
        text_embedding = decoder_output.hidden_states
        text_embedding = self.fc(text_embedding)
        mel_outputs, gate_outputs, alignments = self.tts_decoder.inference(text_embedding)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return mel_outputs, mel_outputs_postnet

# this is 1-scale feature combine with tts-encoder, tts-decoder, postnet
class VIT_LLM_TTS_DecoderV5(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        vit_model = ViT('B_16_imagenet1k', pretrained=True)
        del vit_model.fc
        del vit_model.norm
        self.visual_encoder = vit_model
        med_config = BertConfig.from_json_file(med_config)
        vision_width = 768
        med_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med_config)
        self.text_decoder.apply(weights_init_xavier)
        hparams = create_hparams()

        self.tts_encoder = TTS_Encoder(hparams)
        self.tts_decoder = TTS_Decoder(hparams)
        self.postnet = Postnet(hparams)
        ### Load pretrained tacotron2 model, I trained many times, the performance is so bad
        # checkpoint_path = '/home/ldap-users/s2220411/Code/new_explore/tacotron2/outdir/flickr8k_ResDAVEnet_vq3_v2/checkpoint_100000.pth'
        # cktp = torch.load(checkpoint_path)['state_dict']
        # paramer_tts_encoder = OrderedDict()
        # paramer_tts_decoder = OrderedDict()
        # paramer_tts_post = OrderedDict()
        # for k, v in cktp.items():
        #     if 'encoder' in k:
        #         paramer_tts_encoder[k[8:]] = v
        #     elif 'decoder' in k:
        #         paramer_tts_decoder[k[8:]] = v
        #     elif 'postnet' in k:
        #         paramer_tts_post[k[8:]] = v
        # self.tts_encoder.load_state_dict(paramer_tts_encoder, strict=True)
        # self.tts_decoder.load_state_dict(paramer_tts_decoder, strict=True)
        # self.postnet.load_state_dict(paramer_tts_post, strict=True)
        ### Load pretrained tacotron2 model, I trained many times, the performance is so bad

        self.fc = nn.Linear(1024, 512)

    def forward(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           output_hidden_states=True,
                                           )
        # text_embedding = decoder_output.logits
        text_embedding = decoder_output.hidden_states[-1] # last layer
        text_embedding = self.fc(text_embedding)
        text_embedding = text_embedding.transpose(1, 2)
        encoder_outputs = self.tts_encoder(text_embedding, text_lengths)
        mel_outputs, gate_outputs, alignments = self.tts_decoder(encoder_outputs, mels, memory_lengths=text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return decoder_output, mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def generate(self, image, sample=False, num_beams=5, max_length=40, min_length=30, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = torch.tensor([1024]).unsqueeze(0).to(image.device)
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        return outputs[0].cpu().numpy().tolist()


    def synthesis(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        # text_embedding = decoder_output.logits
        text_embedding = decoder_output.hidden_states
        text_embedding = self.fc(text_embedding)
        text_embedding = text_embedding.transpose(1, 2)
        encoder_outputs = self.tts_encoder.inference(text_embedding)
        mel_outputs, gate_outputs, alignments = self.tts_decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return mel_outputs, mel_outputs_postnet

# this is multi-scale feature combine with tts-encoder, tts-decoder, postnet
class VIT_LLM_TTS_DecoderV6(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        vit_model = ViT('B_16_imagenet1k', pretrained=True)
        del vit_model.fc
        del vit_model.norm
        self.visual_encoder = vit_model
        med_config = BertConfig.from_json_file(med_config)
        vision_width = 768
        med_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med_config)
        self.text_decoder.apply(weights_init_xavier)
        hparams = create_hparams()

        self.tts_encoder = TTS_Encoder(hparams)
        self.tts_decoder = TTS_Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.fc = nn.Linear(1024, 512)

    def forward(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           output_hidden_states=True,
                                           )
        # text_embedding = decoder_output.logits
        hidden_states = decoder_output.hidden_states # len = number of layers + 1, but first layer is bert embedding
        hidden_states = hidden_states[1:]
        # plus all hidden states --> multi-scale feature
        hidden_states = torch.sum(torch.stack(hidden_states), dim=0)
        text_embedding = self.fc(hidden_states)
        text_embedding = text_embedding.transpose(1, 2)
        encoder_outputs = self.tts_encoder(text_embedding, text_lengths)
        mel_outputs, gate_outputs, alignments = self.tts_decoder(encoder_outputs, mels, memory_lengths=text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return decoder_output, mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def generate(self, image, sample=False, num_beams=5, max_length=40, min_length=30, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = torch.tensor([1024]).unsqueeze(0).to(image.device)
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=1025,
                                                 pad_token_id=0,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        return outputs[0].cpu().numpy().tolist()


    def synthesis(self, image, labels, decoder_target, attention_mask, text_lengths, mels, mel_lengths):
        image_embeds = self.visual_encoder(image)  # (batch_size, 196, 768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        decoder_output = self.text_decoder(labels,
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_target,
                                           return_dict=True,
                                           )
        # text_embedding = decoder_output.logits
        text_embedding = decoder_output.hidden_states
        text_embedding = self.fc(text_embedding)
        text_embedding = text_embedding.transpose(1, 2)
        encoder_outputs = self.tts_encoder.inference(text_embedding)
        mel_outputs, gate_outputs, alignments = self.tts_decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return mel_outputs, mel_outputs_postnet