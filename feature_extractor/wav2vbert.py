
import torch

from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.data import Collater
from pathlib import Path
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model
import time

import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

from transformers import (
    AutoFeatureExtractor,
    AutoModel,
)

import utils
import torch.nn as nn

class W2VBert(nn.Module):
    def __init__(self):
        super().__init__()
        base_path = "facebook/w2v-bert-2.0"
        self.model = AutoModel.from_pretrained(base_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(base_path)
    def forward(self, x):
        input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats



def get_model(device, dtype):
    audio_decoder = AudioDecoder(dtype=torch.float32, device=device)
    fbank_converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2 ** 15,
        channel_last=True,
        standardize=True,
        device=device,
        dtype=dtype,
    )
    collater = Collater(pad_value=1)

    model = load_conformer_shaw_model("conformer_shaw", device=device, dtype=dtype)
    model.eval()
    return audio_decoder,fbank_converter,collater,model

def get_content(hmodel, wav_path):
    audio_decoder, fbank_converter, collater, model = hmodel

    with Path(wav_path).open("rb") as fb:
        block = MemoryBlock(fb.read())

    decoded_audio = audio_decoder(block)
    t = time.time()
    src = collater(fbank_converter(decoded_audio))["fbank"]
    seqs, padding_mask = get_seqs_and_padding_mask(src)

    with torch.inference_mode():
        seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
        seqs, padding_mask = model.encoder(seqs, padding_mask)
    return seqs.transpose(1,2)


#
if __name__ == '__main__':
    model = get_model(torch.device('cpu'), torch.float32)
    src_path = "/Users/xingyijin/Downloads/Havana_1d_4x_1channel_vae.wav"
    feats = get_content(model,src_path)
    print(feats.shape)

