import os.path
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from matplotlib import pyplot as plt


phoneme_path = 'dump/phoneme.npy'
phoneme_data = np.load(phoneme_path, allow_pickle=True).item()
print(len(phoneme_data))
hz=50
def generate_data():
    for k, v in phoneme_data.items():
        itemname = k

        code_path = itemname.replace(".wav", ".code.pt")
        if not os.path.exists(code_path):
            continue
        code = torch.load(code_path).squeeze(0)

        dur = code.shape[-1] / hz
        spk_emb_path = itemname.replace(".wav", ".spk.npy")
        if not os.path.exists(spk_emb_path):
            print( f'spk_emb_path {spk_emb_path} not exists for {itemname}\n\n\n\n\n\n\n\n\n')
            continue
        spk_emb = np.load(spk_emb_path)

        if spk_emb.shape != (256, ):
            print( f'spk_emb shape {spk_emb.shape} not correct for {itemname}\n\n\n\n\n\n\n\n\n')
        if dur > 100:
            print(f'dur {dur} too long for {itemname}\n\n\n\n\n\n\n\n')
        yield {"item_name": itemname, "codes": code, 'phoneme': v, 'length': dur, 'spk_emb': spk_emb}

dataset = Dataset.from_generator(generate_data)
print(dataset)

test_samples = 10

split_dataset = dataset.train_test_split(test_size=test_samples/len(dataset), shuffle=True)

split_dataset.save_to_disk('dump/phone_semantic_dataset')
print("average duration", np.mean(dataset['length']))

