import os
import sys

# HiFi 디렉토리를 Python 경로에 추가
sys.path.append("./HiFi")

from HiFi.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
import os
import json
import numpy as np
import torch
from scipy.io.wavfile import read

# JSON 파일에서 설정값 불러오기
config_path = './HiFi/config_v1.json'  # config 파일 경로
with open(config_path, 'r') as f:
    config = json.load(f)

# 설정값 할당
n_fft = config['n_fft']
num_mels = config['num_mels']
sampling_rate = config['sampling_rate']
hop_size = config['hop_size']
win_size = config['win_size']
fmin = config['fmin']
fmax = config['fmax']

# WAV 파일을 불러오는 함수
def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

# WAV 파일을 멜 스펙트로그램으로 변환하고 .npy 파일로 저장하는 함수
def save_mel_spectrogram(wav_path, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # WAV 파일 로드

    audio, sr = load_wav(wav_path)
    audio = audio / MAX_WAV_VALUE
    audio = torch.FloatTensor(audio).unsqueeze(0)

    # 멜 스펙트로그램 계산
    mel_spec = mel_spectrogram(audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
    mel_spec = mel_spec.squeeze(0).cpu().numpy()

    # 저장할 .npy 파일 경로
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(wav_path))[0] + '.npy')

    # .npy 파일로 저장
    np.save(output_file, mel_spec)
    print(f"Saved mel spectrogram to {output_file}")

# 실행 예시
wav_file_path = './music_test/AM Contra - Heart Peripheral/bass_11.wav'   # 처리할 .wav 파일 경로
output_directory = './npy_test'  # 저장할 폴더 경로
save_mel_spectrogram(wav_file_path, output_directory)
