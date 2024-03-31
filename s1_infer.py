import json
from pyannote.audio import Model
from pyannote.audio import Inference

import time
import torch

from mq.trainer import Wav2TTS
from text import cleaned_text_to_sequence
from text.cleaner import text_to_sequence, clean_text
from s2_infer import decode_to_file, encode_from_file
from gen_phonemes import get_bert_feature

text = "当然,不同问题之间错综复杂,对应的结论也有冲突.所以我想要的是'平衡',也就是在所有问题中找到一个'最优解'."
# text = "当然,不同问题之间错综复杂."
# text = "当然,不同问题之间错综复杂,对应的结论也有冲突."
# text= "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。"

prompt_wav_path = "dataset_raw/zh/Nana7mi/Nana7mi_488.wav"
device = torch.device('cuda')
ckpt_path = 'logs/s1/last.ckpt'


def text2phoneid(text, lang='zh'):
    phones, word2ph, norm_text = clean_text(text, lang)
    print(phones)
    bert = get_bert_feature(norm_text, word2ph, 'cpu', lang)
    return phones, cleaned_text_to_sequence(phones), bert


# phlist, phones, bert = text2phoneid(prompt_text+text)
phlist, phones, bert = text2phoneid(text)

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")
spk_emb = inference(prompt_wav_path)
# prompt = encode_from_file(prompt_wav_path)
# prompt_len = prompt.shape[-1]
# prompt_len = int((prompt_len//2)*2)
# prompt = prompt[:,:,:prompt_len].transpose(-1, -2).reshape(1, -1, 8)
prompt = None
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


args = AttrDict()

arg_dict = json.load(open('configs/s1.json', 'r'))
args.__dict__.update(arg_dict)
model = Wav2TTS(args).to(device)
state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
model.load_state_dict(state_dict, strict=True)
model.eval()

total = sum([param.nelement() for param in model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))

all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
spk_emb = torch.FloatTensor(spk_emb).to(device).unsqueeze(0)
print(all_phoneme_ids.shape)
all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
bert = bert.to(device).unsqueeze(0)

st = time.time()
with torch.no_grad():
    model.hp.sampling_temperature = 1
    model.hp.top_p = 0.02
    model.hp.repetition_penalty = 1.5
    pred_semantic = model.inference(
        all_phoneme_ids,
        spk_emb,
        prompt=prompt,
        bert=bert,
        cfg_scale=1.2
    )

print(pred_semantic[0].shape)
print(f'{time.time() - st} sec used in T2S')

out = pred_semantic[0].reshape(-1,4).T.unsqueeze(0)
decode_to_file(out, prompt_wav_path, "tmp.wav")
