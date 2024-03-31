import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .transformers import TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder, CrossAttnOnlyLayer, AlibiPostionEmbedding
from .transducer import Transducer
import numpy as np
import statistics

class TTSDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                [TransformerEncoderLayer(hp) for i in range(hp.enc_nlayers)]
            )
        )
        self.decoder = TransformerDecoder(
            nn.ModuleList(
                [TransformerDecoderLayer(hp, with_cross_attention=True) for i in range(hp.dec_nlayers)]
            )
        )
        self.layer_norm_phone = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.layer_norm_spkr = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.transducer = Transducer(hp)
        self.alibi = AlibiPostionEmbedding(hp.nheads, 10000)
        self.layer_norm = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.tgt_mask = (torch.tril(torch.ones(10000, 10000), diagonal=0) == 0)

    def forward(self, q,q_tgt, phone, spkr, q_mask, phone_mask):
        #Fused phone + speaker
        ex_phone_mask = torch.cat([torch.zeros((spkr.size(0), 1), device=spkr.device, dtype=torch.bool),
                                   phone_mask], 1) if phone_mask is not None else None
        spkr = self.layer_norm_spkr(spkr.unsqueeze(1))
        phone = self.layer_norm_phone(phone)
        phone = torch.cat([spkr, phone], 1)
        phone_alibi = self.alibi(phone)
        phone_alibi[:, 0] = 0
        phone_alibi[:, :, 0] = 0
        phone, enc_attn = self.encoder(phone, mask=None, attn_bias=phone_alibi, src_key_padding_mask=ex_phone_mask)
        phone = phone[:, 1:]
        #Run decoder
        q_mask = torch.cat([torch.zeros((spkr.size(0), 1), device=spkr.device, dtype=torch.bool),
                            q_mask], 1) if q_mask is not None else None
        q_input = q
        q = self.transducer.encode(q)
        q = self.layer_norm(q)
        q = torch.cat([spkr, q], 1)
        tgt_len = q.size(1)
        tgt_mask = self.tgt_mask[: tgt_len, : tgt_len].to(q.device)
        audio_alibi = self.alibi(q)
        audio_alibi[:, 0] = 0
        audio_alibi[:, :, 0] = 0
        output, _, dec_attn, _ = self.decoder(q, memory=phone,
                                              tgt_mask=tgt_mask,
                                              attn_bias=audio_alibi,
                                              tgt_key_padding_mask=q_mask,
                                              memory_key_padding_mask=phone_mask)
        audio_output = output[:, 1:]
        audio_output = self.transducer.decode(audio_output, q_tgt)
        return {
            'logits': audio_output,
            'decoder_attention': dec_attn,
            'encoder_attention': enc_attn
        }

    def encode_phone(self, phone, spkr, phone_mask):
        phone = self.layer_norm_phone(phone)
        phone = torch.cat([spkr, phone], 1)
        ex_phone_mask = torch.cat([torch.zeros((spkr.size(0), 1), device=spkr.device, dtype=torch.bool), phone_mask], 1)
        phone_alibi = self.alibi(phone)
        phone_alibi[:, 0] = 0
        phone_alibi[:, :, 0] = 0
        phone, enc_attn = self.encoder(phone, mask=None, attn_bias=phone_alibi, src_key_padding_mask=ex_phone_mask)
        phone = phone[:, 1:]
        return phone

    def inference_topkp_sampling_batch(self, phone, spkr, phone_mask, prior=None,cfg_scale=1, **kwargs):
        batch_size = phone.size(0)
        spkr = self.layer_norm_spkr(spkr.unsqueeze(1))
        inp = self.layer_norm(self.transducer.start_token(phone.device)) #1, 1, C
        inp = inp.expand(batch_size, -1, -1) #N, 1, C
        inp = torch.cat([spkr, inp], 1)
        prior_size = 0
        if prior is not None:
            prior = self.transducer.encode(prior)
            prior = self.layer_norm(prior)
            prior_size = prior.size(1)
            inp = torch.cat([inp, prior], 1)
        phone = self.encode_phone(phone, spkr, phone_mask)
        tgt_mask = self.tgt_mask[:inp.size(1), :inp.size(1)].to(inp.device)
        inps = inp
        #Decode
        past_kvs1, past_kv_cross, past_kvs2, clusters = None, None, None, torch.empty([batch_size, 0, self.hp.n_cluster_groups], device=phone.device, dtype=torch.long)
        audio_alibi = self.alibi(inp)
        audio_alibi[:, 0] = 0
        audio_alibi[:, :, 0] = 0
        for i in tqdm(range(self.hp.max_output_length)):
            phone_uncond = torch.zeros_like(phone).to(phone.device)
            batched_phone = torch.cat([phone, phone_uncond], 0)
            batched_inps = torch.cat([inps, inps], 0)
            batched_phone_mask = torch.cat([phone_mask, phone_mask], 0)
            cond, _, _, _ = self.decoder(batched_inps, memory=batched_phone, attn_bias=audio_alibi, tgt_mask=tgt_mask, past_kvs=None,
                                              memory_key_padding_mask=batched_phone_mask)
            cond = cond[:, -1].unsqueeze(1) #N, 1, C
            #Run sub-decoder inference
            output = []
            for j in range(self.hp.n_cluster_groups):
                q_input = torch.cat(output, 1) if j else None
                batched_q_input = torch.cat([q_input, q_input], 0) if q_input is not None else None
                batched_logits = self.transducer.decoder.infer(cond, batched_q_input) #N, n_codes
                logit_cond, logit_uncond = torch.chunk(batched_logits, 2, 0)

                logit = logit_uncond + (logit_cond - logit_uncond) * cfg_scale
                #Block Start Token
                logit[:, self.hp.n_codes + 1] = -float("Inf")
                #Repetition penalty
                if self.hp.use_repetition_token and self.hp.repetition_penalty != 1.0:
                    logit[:, self.hp.n_codes + 2] /= self.hp.repetition_penalty
                if self.hp.use_repetition_gating:
                    logit[:, self.hp.n_codes + 2] = torch.min(torch.max(logit[:, :self.hp.n_codes]), logit[:, self.hp.n_codes + 2])
                #Top_p
                if self.hp.top_p < 1.0 and self.hp.top_p > 0.0:
                    sorted_logits, sorted_idxs = torch.sort(logit, descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    idx_to_remove = cum_probs > self.hp.top_p
                    idx_to_remove[:, :self.hp.min_top_k] = False
                    idx_to_remove = idx_to_remove.scatter(1, sorted_idxs, idx_to_remove)
                    logit[idx_to_remove] = -float("Inf")
                #Sampling
                probs = torch.softmax(logit / self.hp.sampling_temperature, dim=-1)
                idx = torch.multinomial(probs, 1) #N, 1
                #If is repetition token
                if self.hp.use_repetition_token:
                    if clusters.size(1) == 0: #First token, random choice
                        idx[idx==(self.hp.n_codes + 2)] = torch.randint(self.hp.n_codes, size=(1,), device=idx.device)
                    else:
                        idx[idx==(self.hp.n_codes + 2)] = clusters[:, -1:, j][idx==(self.hp.n_codes + 2)]
                output.append(idx)
            output = torch.cat(output, 1).unsqueeze(1) #N, 1, n_groups

            if self.transducer.is_end_token_batch(output):
                break
            if i == self.hp.max_output_length - 1:
                break

            #Update args
            tgt_mask = self.tgt_mask[:i+3+prior_size, :i+3+prior_size].to(phone.device)
            audio_alibi = self.alibi(tgt_mask)
            audio_alibi[:, :, 0] = 0

            inp = self.transducer.encode(output)
            inp = self.layer_norm(inp)
            inps = torch.cat([inps, inp], 1)
            clusters = torch.cat([clusters, output], 1) #N, T, 4
        return clusters
