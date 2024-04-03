# AutoRegressive-VITS

> MQTTS branch

(WIP) text to speech using autoregressive transformer and VITS 
## Note
+ 此分支为AR-VITS的多码本+MQTTS分支，用于实验多码本解码，[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的单码本版本在[master分支](https://github.com/innnky/ar-vits/tree/master) 
+ 效果一般，实验性分支，而且pretrain规模较小（zh-300h ja-80h en-20h），语言较少，效果不及[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 
+ 没有做推理工程化加速，推理速度极极极极极慢，仅供实验使用
+ 由于底模数据原因模型基本只有中文能力，而同样由于训练集时长均较短，模型只能合成较短的句子，长句需要切片分开推理，否则会爆炸。
+ 已测试过的微调配置：30分钟数据+s2微调1200步+s1微调100步 效果 -> [sample](https://huggingface.co/innnky/ar-vits/blob/main/samples/%E4%B8%83%E6%B5%B7%E5%87%BA%E5%B8%88%E8%A1%A8.wav) 微调数据来自[Xz乔希](https://www.bilibili.com/video/BV1KA4m1V71D)
+ 需要指定参考音频，但此分支使用的是声纹embedding，而非prompt的方式，因此参考音频参考效果不是很强
+ 所有脚本只在linux下测试通过，未在win下测试
+ 如果无法连接huggingface下载bert、hubert等模型，建议使用`export HF_ENDPOINT=https://hf-mirror.com`


## Acknowledgement
+ Thanks to the support of the GPUs by [leng-yue](https://github.com/leng-yue)

## Reference
+ autoregressive mqtts transformer from [MQTTS](https://github.com/b04901014/MQTTS)
+ VITS from [VITS](https://github.com/jaywalnut310/vits)
+ reference encoder from [TransferTTS](https://github.com/hcy71o/TransferTTS)

## Training pipeline
1. jointly train S2 vits decoder and quantizer
2. extract semantic tokens
3. train S1 text to semantic
## preparation
+ download pretrained models
```shell
bash download_pretrain.sh
```
+ put training data in `dataset_raw` folder with the following structure
```
dataset_raw
├── zh
│   ├── spk1
│   │   ├── utt1.wav
│   │   ├── utt1.lab
│   │   ├── ...
│   ├── spk2
│   │   ├── utt1.wav
│   │   ├── utt1.lab
│   │   ├── ...
```
## vits S2 training
+ resample.py
+ gen_phonemes.py
+ extract_ssl_s2.py
+ gen_filelist_s2.py
+ s2_train.py
```shell
python s2_train.py -c configs/s2.json -p pretrain/s2
```

## mqtts S1 training
+ extract_vq_s1.py
+ extract_spk_embedding.py
+ gen_filelist_s1.py
+ s1_train.py
```shell
python s1_train.py
```
## Inference
+ s1_infer.py/s2_infer.py
