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

    def __init__(self, meta_file, wav_dir, batch_size=32, do_bucket=False, bucket_size=20):
        meta = pd.read_csv(meta_file, sep='|', header=None)
        self.batch_size = batch_size
        self.do_bucket = do_bucket
        self.bucket_size = bucket_size 
        self.wav_dir = wav_dir
        self.wav_list = meta[0].tolist()
        txt_origin = meta[1].tolist()
        txt_prepro = meta[2].tolist()
        self.txt_list = self._make_text_list(txt_origin, txt_prepro)
        self._remove_long_text()
        self.n_example = len(self.txt_list)
        # Bucketing
        if self.do_bucket:
            self.b, self.b_map, self.b_boundaries = self._sort_and_bucket()
            
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

    def _sort_and_bucket(self):
        # Old length list
        L = [len(self.txt_list[i]) for i in range(self.n_example)]
        # Sort
        idx = sorted(range(self.n_example), key=lambda i: L[i])
        self.txt_list = [self.txt_list[i] for i in idx]
        self.wav_list = [self.wav_list[i] for i in idx]
        # New length list
        L = [len(self.txt_list[i]) for i in range(self.n_example)]
        # Bucket
        bucket_boundaries = [(k * self.bucket_size, (k+1) * self.bucket_size) for k in range(math.ceil(max(L) / self.bucket_size))]
        n_bucket = len(bucket_boundaries)
        bucket_map = []
        bucket = [[] for _ in range(n_bucket)]
        for i, l in enumerate(L):
            item = {'wav': self.wav_list[i], 'text': self.txt_list[i]}
            # Search which bucket to put item
            if l == bucket_boundaries[-1][1]:
                bucket_map.append(n_bucket-1)
                bucket[-1].append(item)
            else:
                for k in range(n_bucket):
                    if bucket_boundaries[k][0] <= l < bucket_boundaries[k][1]:
                        bucket_map.append(k)
                        bucket[k].append(item)
            
        return bucket, bucket_map, bucket_boundaries


    def _make_example(self, wav_name, text):
        wav_file = os.path.join(self.wav_dir, wav_name + '.wav')
        wav = load_audio(wav_file)
        mel, mag = get_spectrogram(wav)
        return {'text': text, 'mel': mel, 'mag': mag}

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        if self.do_bucket:
            # Pick idx-th sample and another batch_size-1 samples
            wav = self.wav_list[idx]
            txt = self.txt_list[idx]
            b_id = self.b_map[idx]
            n_sample = self.batch_size-1 if self.batch_size <= len(self.b[b_id]) else len(self.b[b_id])-1
            samples = random.sample(self.b[b_id], k=n_sample)
            samples += [{'wav': wav, 'text': txt}]
            batch = [self._make_example(samples[i]['wav'], samples[i]['text'])
                     for i in range(len(samples))]
            item = collate_fn(batch)
        else:
            item = self._make_example(self.wav_list[idx], self.txt_list[idx])
        return item


def collate_fn(batch):
    #GO_frame = np.zeros([1, hps.n_mels])
    # Add ending token at the end
    idx = [sent2idx(b['text']) + [hps.char_set.find('E')] for b in batch]
    # Add GO frame at the beginning
    #mel = [np.concatenate([GO_frame, b['mel']], axis=0) for b in batch]
    mel = [b['mel'] for b in batch]
    mag = [b['mag'] for b in batch]

    max_text_len = max([len(x) for x in idx])
    max_time_step = max([x.shape[0] for x in mel])
    # for reduction factor
    remain = max_time_step % hps.reduction_factor
    max_time_step += (hps.reduction_factor - remain)

    # Padding
    for i, x in enumerate(idx):
        L = len(x)
        diff = max_text_len - L
        pad = [hps.char_set.find('P') for _ in range(diff)]
        idx[i] += pad

    for i, x in enumerate(mel):
        L = x.shape[0]
        diff = max_time_step - L
        pad = np.zeros([diff, x.shape[1]])
        mel[i] = np.concatenate([x, pad], axis=0)

    for i, x in enumerate(mag):
        L = x.shape[0]
        diff = max_time_step - L
        pad = np.zeros([diff, x.shape[1]])
        mag[i] = np.concatenate([x, pad], axis=0)

    return {'text': torch.LongTensor(idx),
            'mel': torch.Tensor(mel),
            'mag': torch.Tensor(mag)}


