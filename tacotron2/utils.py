import numpy as np
from scipy.io.wavfile import read
import torch
import os
import json
import hifigan

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def load_vocoder():
    # path_script = "/home/ldap-users/s2220411/Code/new_explore/tacotron2"
    path_script = os.path.dirname(os.path.abspath(__file__))
    path_script = os.path.join(path_script, "..")
    with open(os.path.join(path_script, "hifigan/config.json"), "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load(os.path.join(path_script, "hifigan/generator_universal.pth.tar"))
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to("cuda")
    return vocoder

def vocoder_infer(mels, vocoder, lengths=None):
    wavs = vocoder(mels).squeeze(1)
    max_value = 32766
    wavs = (wavs.cpu().numpy()* max_value).astype("int16")
    return wavs