# Packet Loss Concealment

Carnival system을 구성하는 packet loss concealment (PLC) 모델입니다. 과학기술통신부 재원으로 정보통신기획평가원(IITP) 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제 공개 코드입니다. (2021.05~2024.12)

본 PLC 모델은 딥러닝 기반의 은닉 패킷 손실 보상 방법으로 EID error로 인해 발생한 패킷 손실을 보상해주는 방법으로 구성되어 있습니다. 본 실험은 SiTEC 한국어 음성 DB를 사용하여 진행되었습니다.

Deep neural network 모델을 기반으로 음성신호처리에 중요한 정보인 pitch 정보를 추가하여 모델의 성능을 개선하였습니다.

Done
 - Deep neural networks 기반의 PLC 베이스라인 시스템 구축
 - 각 음성의 pitch 정보를 추출하여 DNN 모델에 conditioning으로 활용
 - Time-frequence 도메인 기반의 모델에 time 도메인에서 주로 사용하는 SI-SNR loss 추가


To do
 - Recurrent 모델 기반 시스템으로 변경
 - Time 도메인 기반 모델로 변경
 - 음성신호처리에 중요한 다른 정보의 conditioning

## Dependencies

- Python3
- torch==1.1.0
- librosa==0.7.0
- SoundFile==0.10.2
- tensorboard==1.14.0
- tensorboard==1.13.1(for visualization only)
- matplotlib==3.1.0
- tqdm==4.32.2

## Training

To train the baseline model run this command:
```
		python train_baseline.py
```
To train the model utilizing pitch information, run this command:
```
		python train_pitch.py
```
you can change settings at config/train/train_*.json
