import sys

import librosa
import soundfile
import torch
import torchaudio

import utils
from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch, spec_to_mel_torch
from feature_extractor.cnhubert import get_model, get_content
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

models = None


def load_model(device="cuda", config_path="configs/s2.json", model_path=None, skip_ssl=False):
    global models
    device = torch.device(device)
    if models is not None:
        return models
    print('loading models...')
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    if model_path is None:
        model_path = utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth")
    utils.load_checkpoint(model_path, net_g,
                          None, False)
    net_g.eval()
    ssl = get_model().to(device)
    models = (hps, net_g, ssl)
    return models


def get_spepc(hps, filename):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    audio = audio.unsqueeze(0)
    if sampling_rate != hps.data.sampling_rate:
        audio = torchaudio.functional.resample(audio, sampling_rate, hps.data.sampling_rate)
    audio_norm = audio
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,
                             hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                             center=False)
    return spec



@torch.no_grad()
@torch.inference_mode()
def decode_to_file(codes, ref_path, save_path):
    device = codes.device
    hps, net_g, ssl = load_model(device=device)
    ref = get_spepc(hps, ref_path).to(device)

    audio = net_g.decode(codes.transpose(0, 1), ref).detach().cpu().numpy()[0, 0]
    soundfile.write(save_path, audio, hps.data.sampling_rate)


@torch.no_grad()
@torch.inference_mode()
def encode_from_file(path, device='cpu', config_path="configs/s2.json", model_path=None):
    hps, net_g, ssl = load_model(device=device, config_path=config_path, model_path=model_path)
    ssl_content = get_content(ssl, path).to(device)
    codes = net_g.extract_latent(ssl_content)
    return codes


@torch.no_grad()
@torch.inference_mode()
def encode_ge_from_file(path, device='cpu', config_path="configs/s2.json", model_path=None):
    hps, net_g, ssl = load_model(device=device, config_path=config_path, model_path=model_path)
    ref = get_spepc(hps, path).to(device)
    ge = net_g.extract_ge(ref)
    return ge


if __name__ == '__main__':
    refer_path = "/home/fish/genshin_data/zh/钟离/vo_ZLLQ104_CS_zhongli_04.wav"
    src_path = "/home/fish/genshin_data/zh/派蒙/vo_ABDLQ001_1_paimon_01.wav"
    refer_path = src_path
    # src_path=  refer_path
    device = 'cpu'
    codes = encode_from_file(src_path, device=device)
    print(codes.shape)
    decode_to_file(codes, refer_path, "tmp.wav")
    # infer_test(src_path)