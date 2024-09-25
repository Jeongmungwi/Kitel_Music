import torch
import torchaudio
import numpy as np
import sys
import json

sys.path.append("./HiFi")
from HiFi.meldataset import mel_spectrogram
from block import UNet

# CUDA 또는 CPU 자동 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 설정 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# 오디오 파일 로드 함수
def load_wav_to_tensor(wav_path, sampling_rate):
    waveform, sr = torchaudio.load(wav_path)
    if sr != sampling_rate:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        waveform = resample(waveform)
    return waveform

# 모델 로드 함수
def load_model(checkpoint_path, config):
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # 모델을 평가 모드로 전환 (추론을 위해)
    return model

# Mel 스펙트로그램 예측 및 저장 함수
def predict_mel_spectrogram(model, wav_path, config, output_npy_path):
    # 오디오 파일 로드
    mixture_wav = load_wav_to_tensor(wav_path, config['sampling_rate']).to(device)

    # Mel 스펙트로그램 생성
    mixture_mel = mel_spectrogram(mixture_wav,
                                  config['n_fft'],
                                  config['num_mels'],
                                  config['sampling_rate'],
                                  config['hop_size'],
                                  config['win_size'],
                                  config['fmin'],
                                  config['fmax']).to(device)

    # 입력 데이터 차원 추가 (batch_size, channels, height, width)
    mixture_mel = mixture_mel.unsqueeze(0)

    # 모델 예측
    with torch.no_grad():  # 추론 시에는 gradient 계산을 하지 않음
        predicted_bass_mel = model(mixture_mel)
    
    # batch 및 channel 차원 제거
    predicted_bass_mel = predicted_bass_mel.squeeze(0).squeeze(0).cpu().numpy()

    # Mel 스펙트로그램을 .npy 파일로 저장
    np.save(output_npy_path, predicted_bass_mel)
    print(f"Predicted Mel Spectrogram saved at: {output_npy_path}")

def main():
    # 설정 파일 경로 및 모델 체크포인트 경로
    config_path = './HiFi/config_v1.json'
    checkpoint_path = './ck_bass/unet_epoch_30.pth'  # 예시로 10번째 에포크에서 저장된 모델
    
    # 설정 로드
    config = load_config(config_path)

    # 모델 로드
    model = load_model(checkpoint_path, config)

    # 테스트할 .wav 파일 경로와 저장할 .npy 파일 경로
    wav_path = './music_test/AM Contra - Heart Peripheral/mixture_11.wav'  # 테스트할 .wav 파일 경로
    output_npy_path = './npy_test/bass_30.npy'  # 저장할 .npy 파일 경로

    # Mel 스펙트로그램 예측 및 저장
    predict_mel_spectrogram(model, wav_path, config, output_npy_path)

# main 함수 실행
if __name__ == "__main__":
    main()
