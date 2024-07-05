# AutoRegressive-VITS

(WIP) 텍스트를 음성으로 변환하는 자동회귀 변환기와 VITS
## 노트
+ 모델의 효과가 완전히 검증되지 않았으므로, 반드시 좋은 성능을 보장하지 않습니다. 신중하게 진행하시기 바랍니다. 사전 훈련된 모델은 아직 훈련 중입니다.
+ 처음부터 훈련하려면 엄청난 양의 데이터가 필요합니다(적어도 수천 시간?) (valle, speartts, soundstorm와 유사). 데이터가 적으면 좋은 결과를 얻을 수 없습니다.
+ VITS+reference는 zeroshot 방향에서 큰 한계가 있기 때문에, 이 저장소는 zeroshot을 목표로 하지 않습니다. 이 저장소의 목표는 큰 언어 모델의 사전 훈련이 있는 경우 자동회귀 언어 모델의 힘을 빌려 소량의 데이터로 미세 조정한 후에도 좋은 운율을 얻는 것입니다.
+ 몇 가지 초기 [합성 샘플](https://huggingface.co/innnky/ar-tts-models/tree/main/gpt-vits)을 간단히 업데이트했습니다.
## 할 일
+ [x] 원신 데이터로 훈련
+ [x] 더 많은 중국어 오픈 소스 데이터를 수집하여 훈련(약 600시간 예상)하고 사전 훈련 모델 공개 (x) --> 분포 외 텍스트의 성능이 매우 나쁘며, 예를 들어 고전 중국어를 읽는 것과 긴 문장에서 효과가 좋지 않아 이상한 행동을 할 수 있습니다.
  + [ ] 분포 외 성능을 개선하기 위해 단어 수준의 BERT를 추가하고 음소 수준으로 반복
  + [ ] 같은 화자의 여러 데이터를 하나의 오디오 파일로 결합하여 평균 데이터 길이를 늘리고 긴 문장 합성의 안정성을 향상
  + [ ] 긴 문장 합성의 안정성을 개선하기 위해 RoPE 상대 위치 인코딩으로 변경?
+ [ ] 미세 조정 관련 코드를 작성하고 화자 ID 지원 추가
+ [ ] 일본어와 영어 텍스트 프론트엔드를 최적화하고 더 많은 일본어와 영어 데이터를 수집(각 언어당 약 600시간 예상)하여 훈련하고 사전 훈련 모델 공개

## 구조
![structure.png](resources%2Fstructure.png)

+ [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)의 디코더 전용 텍스트-의미 체계
+ [VITS](https://github.com/jaywalnut310/vits)에서 가져온 VITS
+ [TransferTTS](https://github.com/hcy71o/TransferTTS)에서 가져온 참조 인코더

## 훈련 파이프라인
1. S2 VITS 디코더와 양자화를 공동 훈련
2. 의미 토큰 추출
3. S1 텍스트를 의미로 변환하는 훈련

## VITS S2 훈련
+ resample.py
+ gen_phonemes.py
+ extract_ssl_s2.py
+ gen_filelist_s2.py
+ train_s2.py

## GPT S1 훈련
+ extract_vq_s1.py
+ gen_filelist_s1.py
+ train_s1.py

## 추론
+ s1_infer.py/s2_infer.py (진행 중)

## 사전 훈련된 모델
+ 진행 중
