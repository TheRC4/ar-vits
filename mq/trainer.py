import torch
import torch.nn as nn
import torch.optim as optim
from mq.modules.wildttstransformer import TTSDecoder
import pytorch_lightning.core.module as pl
import matplotlib.pyplot as plt

from text.symbols import symbols

plt.switch_backend('agg')

class Wav2TTS(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.TTSdecoder = TTSDecoder(hp)
        self.n_decode_codes = self.TTSdecoder.transducer.n_decoder_codes
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.hp.label_smoothing)
        self.phone_embedding = nn.Embedding(len(symbols), hp.hidden_size, padding_idx=symbols.index('_'))
        self.spk_proj = nn.Linear(256, hp.hidden_size)
        self.bert_proj = nn.Linear(1024, hp.hidden_size)

    def load(self):
        print(f"Loading pretrained model from {self.hp.pretrained_path}")
        state_dict = torch.load(self.hp.pretrained_path)['state_dict']
        self.load_state_dict(state_dict, strict=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW([*self.parameters()], lr=self.hp.lr, betas=(0.9, 0.95), weight_decay=0.1)
        return optimizer

    def training_step(self, batch, batch_idx):
        acc, loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        acc, loss = self.step(batch)
        self.log("val/loss", loss, on_step=True, prog_bar=True)
        self.log("val/acc", acc, on_step=True, prog_bar=True)

    def step(self, batch):
        speaker_embedding = self.spk_proj(batch['spk_emb'])
        # Deal with phone segments
        phone_features = self.phone_embedding(batch['phone'])
        bert_feature = batch['bert_feature']
        bert_features = self.bert_proj(bert_feature.transpose(1, 2))
        phone_features = phone_features + bert_features
        # Run decoder
        recons_segments = self.TTSdecoder(batch['tts_quantize_input'], batch['tts_quantize_output'], phone_features, speaker_embedding,
                                          batch['quantize_mask'], batch['phone_mask'])
        target = recons_segments['logits'][~batch['quantize_mask']].view(-1, self.n_decode_codes)
        labels = batch['tts_quantize_output'][~batch['quantize_mask']].view(-1)
        loss = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        return acc, loss

    @torch.inference_mode()
    @torch.no_grad()
    def inference(self, phonemes, spk_emb, prompt, bert, cfg_scale=1):
        speaker_embedding = self.spk_proj(spk_emb)
        phone_features = self.phone_embedding(phonemes)
        bert_features  = self.bert_proj(bert.transpose(1, 2))
        phone_features = phone_features + bert_features
        phone_mask = torch.full((phone_features.size(0), phone_features.size(1)), False, dtype=torch.bool, device=phone_features.device)
        output = self.TTSdecoder.inference_topkp_sampling_batch(phone_features, speaker_embedding, phone_mask, prior=prompt, output_alignment=True, cfg_scale=cfg_scale)
        return output