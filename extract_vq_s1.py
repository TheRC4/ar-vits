import math
import multiprocessing
import os
from random import shuffle

import numpy as np
import torch.multiprocessing as mp

import torch
from glob import glob

import torchaudio
from tqdm import tqdm

import utils
import logging

from data_conf import data_root
from module.models import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)


extract_ssl = False
input_wav_sr = 32000
s2_config_path = "configs/s2.json"
s2_ckpt_dir = 'logs/s2'

def process_one(file_path, hps,net_g, ssl_model, device):
    if extract_ssl:
        from feature_extractor.cnhubert import  get_content
        ssl = get_content(ssl_model, file_path)
    else:
        ssl_path = file_path.replace(".wav", ".ssl.pt")
        if not os.path.exists(ssl_path):
            print(f"Skip {file_path}")
            return
        ssl = torch.load(ssl_path).float().to(device)

    code_path = file_path.replace(".wav", ".code.pt")
    if os.path.exists(code_path):
        print(f"Skip {file_path}")
        return
    codes = net_g.extract_latent(ssl).cpu()
    torch.save(codes, code_path)

def process_batch(filenames):
    print("Loading models ...")
    process_idx = mp.current_process()._identity
    rank = process_idx[0] if len(process_idx) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    print(device)
    hps = utils.get_hparams_from_file(s2_config_path)
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    model_path = utils.latest_checkpoint_path(s2_ckpt_dir, "G_*.pth")
    utils.load_checkpoint(model_path, net_g,
                          None, False)
    net_g.eval()
    if extract_ssl:
        from feature_extractor.cnhubert import get_model, get_content
        ssl = get_model(input_sample_rate=input_wav_sr)
        ssl = ssl.to(device)
    else:
        ssl = None
    print("Loaded .")
    with torch.no_grad():
        for filename in tqdm(filenames):
            process_one(filename, hps, net_g, ssl, device)

in_dir = data_root

if __name__ == "__main__":
    filenames = glob(f"{in_dir}/**/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 2
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
