# AutoRegressive-VITS

(WIP) text to speech using autoregressive transformer and VITS 
## Note
+ The model's effectiveness has not been fully verified, so it may not perform well. Please proceed with caution, as the pre-trained model is still being trained.
+ Training from scratch requires massive amounts of data (at least thousands of hours?) (similar to valle, speartts, soundstorm). Having less data will certainly not yield good results.
+ Due to the significant limitations of VITS+reference in the zeroshot direction, this repository does not aim for zeroshot. The goal of this repository is to leverage the power of autoregressive language models to achieve good prosody when fine-tuning on small datasets, assuming a large language model pre-training.
+ Some preliminary [synthesized samples](https://huggingface.co/innnky/ar-tts-models/tree/main/gpt-vits) have been simply updated.
## Todo
+ [x] Training on Genshin Impact data
+ [x] Collecting more Chinese open-source data for training (expected around 600 hours) and releasing the pre-trained model (x) --> Out-of-distribution text performance is very poor, such as reading classical Chinese, and long sentences do not perform well and can behave erratically.
  + [ ] Add word-level BERT and repeat at the phoneme level to improve out-of-distribution performance.
  + [ ] Combine multiple data entries of the same speaker into one audio file to increase the average data duration and improve the stability of long sentence synthesis.
  + [ ] Switch to RoPE relative position encoding to improve the stability of long sentence synthesis?
+ [ ] Write code related to fine-tuning and add support for speaker IDs.
+ [ ] Optimize Japanese and English text frontend, collect more Japanese and English data (expected 600 hours for each language) for training, and release the pre-trained models.

## Structure
![structure.png](resources%2Fstructure.png)

+ Decoder-only text-to-semantic from [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
+ VITS from [VITS](https://github.com/jaywalnut310/vits)
+ Reference encoder from [TransferTTS](https://github.com/hcy71o/TransferTTS)

## Training pipeline
1. Jointly train S2 VITS decoder and quantizer.
2. Extract semantic tokens.
3. Train S1 text-to-semantic.

## VITS S2 Training
+ resample.py
+ gen_phonemes.py
+ extract_ssl_s2.py
+ gen_filelist_s2.py
+ train_s2.py

## GPT S1 Training
+ extract_vq_s1.py
+ gen_filelist_s1.py
+ train_s1.py

## Inference
+ s1_infer.py/s2_infer.py (work in progress)

## Pretrained models
+ work in progress
