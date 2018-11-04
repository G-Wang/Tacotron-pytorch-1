# ------------------------
# - Author:  Tao, Tu
# - Date:    2018/8/30
# - Description:
#       Create dataset to deal with LJSpeech.
#
# -----------------------
import os
import math
import random
import pandas as pd
import numpy as np
from hyperparams import Hyperparams as hps
from utils import load_audio, get_spectrogram, text_normalize, sent2idx
import torch
from torch.utils.data import Dataset


class LJSpeech_Dataset(Dataset):
    """LJSpeech dataset.
    """
    def __init__(self, meta_file, wav_dir, batch_size=32, do_bucket=False):
        meta = pd.read_csv(meta_file, sep='|', header=None)
        self.batch_size = batch_size
        self.do_bucket = do_bucket
        self.wav_dir = wav_dir
        self.wav_list = meta[0].tolist()
        txt_origin = meta[1].tolist()
        txt_prepro = meta[2].tolist()
        self.txt_list = self._make_text_list(txt_origin, txt_prepro)
        self._remove_long_text()
        self.n_example = len(self.txt_list)
        # Bucketing
        if self.do_bucket:
            self.wav_list, self.txt_list = self._sort()

    def _make_text_list(self, txt_origin, txt_prepro):
        text_list = []
        for i in range(len(txt_prepro)):
            if type(txt_prepro[i]) is str:
                t = txt_prepro[i]
            else:
                t = txt_origin[i]
            text_list.append(text_normalize(t))
        return text_list

    def _remove_long_text(self):
        idx = []
        for i, txt in enumerate(self.txt_list):
            if hps.text_min_length <= len(txt) <= hps.text_max_length:
                idx.append(i)
        self.txt_list = [self.txt_list[i] for i in idx]
        self.wav_list = [self.wav_list[i] for i in idx]

    def _sort(self):
        # Length list
        L = [len(self.txt_list[i]) for i in range(self.n_example)]
        # Sort: long -> short
        idx = sorted(range(self.n_example), key=lambda i: -L[i])
        txt_list = [self.txt_list[i] for i in idx]
        wav_list = [self.wav_list[i] for i in idx]
        return wav_list, txt_list


    def _make_example(self, wav_name, text):
        wav_file = os.path.join(self.wav_dir, wav_name + '.wav')
        wav, _ = load_audio(wav_file, sr=hps.sampling_rate)
        mel, mag = get_spectrogram(wav)
        return {'text': text, 'mel': mel, 'mag': mag}

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        if self.do_bucket:
            if idx + self.batch_size <= self.n_example:
                batch_idx = range(idx, idx + self.batch_size)
            else:
                batch_idx = range(self.n_example - self.batch_size, self.n_example)
            batch = [self._make_example(self.wav_list[j], self.txt_list[j]) for j in batch_idx]
            item = collate_fn(batch)
        else:
            item = self._make_example(self.wav_list[idx], self.txt_list[idx])
        return item


def collate_fn(batch):
    # Add ending token at the end
    text_idx = [sent2idx(b['text']) + [hps.vocab.find('E')] for b in batch]
#    # In order to use torch.nn.utils.rnn.pack_padded_sequence, we sort by text length in  a decreasing order
#    text_len = [len(x) for x in text_idx]
#    idx = sorted(range(len(text_idx)), key=lambda i: -text_len[i])
#    # Sort
#    text_idx = [text_idx[i] for i in idx]
    mel = [b['mel'] for b in batch]
    mag = [b['mag'] for b in batch]

    max_text_len = max([len(x) for x in text_idx])
    max_time_step = max([x.shape[0] for x in mel])
    # for reduction factor
    remain = max_time_step % hps.reduction_factor
    max_time_step += (hps.reduction_factor - remain)
    
    text_len = []
    mel_len = []
    
    # Padding
    for i, x in enumerate(text_idx):
        L = len(x)
        diff = max_text_len - L
        pad = [hps.vocab.find('P') for _ in range(diff)]
        text_idx[i] += pad
        text_len.append(L)

    for i, x in enumerate(mel):
        L = x.shape[0]
        diff = max_time_step - L
        pad = np.zeros([diff, x.shape[1]])
        mel[i] = np.concatenate([x, pad], axis=0)
        mel_len.append(L)

    for i, x in enumerate(mag):
        L = x.shape[0]
        diff = max_time_step - L
        pad = np.zeros([diff, x.shape[1]])
        mag[i] = np.concatenate([x, pad], axis=0)
        

    return {'text': torch.LongTensor(text_idx),
            'mel': torch.Tensor(mel),
            'mag': torch.Tensor(mag),
            'text_length': torch.LongTensor(text_len),
            'frame_length': torch.LongTensor(mel_len)}
            
            


