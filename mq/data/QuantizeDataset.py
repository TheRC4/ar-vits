import os
from torch.utils import data
import torch
import json
import numpy as np
import soundfile as sf
import random
from pathlib import Path
from librosa.util import normalize

import torch.nn.functional as F
from tqdm import tqdm

from text import cleaned_text_to_sequence
from text.symbols import symbols


def random_crop(x, maxseqlen):
    if x.shape[0] >= maxseqlen:
        offset = random.randrange(x.shape[0] - maxseqlen + 1)
        x = x[offset: offset + maxseqlen]
    else:
        offset = 0
    return x, offset


def dynamic_range_compression(x, C=0.3, M=6.5, clip_val=1e-5):
    return (np.log(np.clip(x, a_min=clip_val, a_max=None)) + M) * C


def dynamic_range_decompression(x, C=0.3, M=6.5):
    return np.exp(x / C - M)


class QuantizeDataset(data.Dataset):
    def __init__(self, hp, dataset, hz=50):
        self.hp = hp
        self.dataset = dataset
        l = len(self.dataset)
        print(f'Total {l} examples')
        self.lengths = dataset['length']
        avglen = sum(self.lengths) / len(self.lengths)
        maxlen = max(self.lengths)
        minlen = min(self.lengths)
        print(
            f"Average duration of audio: {avglen} sec, Maximum duration: {maxlen} sec, Minimum duration: {minlen} sec")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        i = int(i)
        row = self.dataset[i]
        dataname = row['item_name']
        phoneme = row['phoneme']
        phonemes = cleaned_text_to_sequence(phoneme.split(' '))
        spk_emb = row['spk_emb']

        quantization = np.array(row['codes']).T  # ..., 4

        l = quantization.shape[0]
        l = int(l // 2) * 2
        quantization = quantization[:l]
        quantization = quantization.reshape(-1, 8)

        start, end = np.full((1, self.hp.n_cluster_groups), self.hp.n_codes + 1, dtype=np.int16), np.full(
            (1, self.hp.n_cluster_groups), self.hp.n_codes, dtype=np.int16)

        quantization_s = np.concatenate([start, quantization.copy()], 0)
        if self.hp.use_repetition_token:
            pad = np.full((1, self.hp.n_cluster_groups), -100, dtype=np.int16)
            np_mask = np.diff(quantization, axis=0, prepend=pad)
            quantization[np_mask == 0] = self.hp.n_codes + 2
        quantization_e = np.concatenate([quantization, end], 0)

        bert_path = dataname.replace('.wav', '.bert.pt')
        phones_length = len(phonemes)
        bert_feature = torch.load(bert_path)
        assert bert_feature.shape[-1] == phones_length
        return quantization_s, quantization_e, phonemes, dataname, spk_emb, bert_feature

    def seqCollate(self, batch):
        output = {
            'phone': [],
            'phone_mask': [],
            'tts_quantize_input': [],
            'tts_quantize_output': [],
            'quantize_mask': [],
            'spk_emb': [],
        }
        # Get the max length of everything
        max_len_q, max_phonelen = 0, 0
        for q_s, q_e, ph, _, spk_emb, bert in batch:
            if len(q_s) > max_len_q:
                max_len_q = len(q_s)
            if len(ph) > max_phonelen:
                max_phonelen = len(ph)
            output['spk_emb'].append(spk_emb)

        bert_padded = torch.FloatTensor(len(batch), 1024, max_phonelen)
        bert_padded.zero_()
        # Pad each element, create mask
        for idx, (qs, qe, phone, _, _, bert) in enumerate(batch):
            # Deal with phonemes
            phone_mask = np.array([False] * len(phone) + [True] * (max_phonelen - len(phone)))
            phone = np.pad(phone, [0, max_phonelen - len(phone)])
            # Deal with quantizations
            q_mask = np.array([False] * len(qs) + [True] * (max_len_q - len(qs)))
            qs = np.pad(qs, [[0, max_len_q - len(qs)], [0, 0]], constant_values=self.hp.n_codes)
            qe = np.pad(qe, [[0, max_len_q - len(qe)], [0, 0]], constant_values=self.hp.n_codes)
            # Aggregate
            output['phone'].append(phone)
            output['phone_mask'].append(phone_mask)
            output['tts_quantize_input'].append(qs)
            output['tts_quantize_output'].append(qe)
            output['quantize_mask'].append(q_mask)
            bert_padded[idx, :, :bert.shape[-1]] = bert

        for k in output.keys():
            output[k] = np.array(output[k])
            if 'mask' in k:
                output[k] = torch.BoolTensor(output[k])
            elif k in ['phone', 'tts_quantize_input', 'tts_quantize_output', 'sid']:
                output[k] = torch.LongTensor(output[k])
            else:
                output[k] = torch.FloatTensor(output[k])

        output['bert_feature'] = bert_padded
        return output


class QuantizeDatasetVal(QuantizeDataset):
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        quantization_s, quantization_e, phonemes, dataname, spk_emb, bert = super().__getitem__(i)
        audio, sampling_rate = sf.read(dataname)
        audio = normalize(audio) * 0.95
        return (
            torch.LongTensor(quantization_s),
            torch.LongTensor(quantization_e),
            torch.LongTensor(phonemes),
            torch.FloatTensor(audio),
            torch.FloatTensor(spk_emb),
            bert,
        )
