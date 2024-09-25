import torch
import torchaudio
import json
import os
import sys
from tqdm import tqdm

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

# 데이터셋 폴더에서 파일 경로를 로드하는 함수
def get_audio_file_pairs(data_dir):
    audio_pairs = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            mixture_files = sorted([f for f in os.listdir(folder_path) if f.startswith("mixture_") and f.endswith(".wav")])
            bass_files = sorted([f for f in os.listdir(folder_path) if f.startswith("bass_") and f.endswith(".wav")])  # drums -> bass로 변경
            for mix_file, bass_file in zip(mixture_files, bass_files):
                mixture_path = os.path.join(folder_path, mix_file)
                bass_path = os.path.join(folder_path, bass_file)
                audio_pairs.append((mixture_path, bass_path))
    return audio_pairs

# 모델 학습 함수
def train_model(data_dir, model, optimizer, criterion, config, num_epochs=15, save_dir='./model_checkpoints', log_file='training_log.txt'):
    audio_pairs = get_audio_file_pairs(data_dir)

    # 체크포인트 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 로그 파일 열기
    with open(log_file, 'w') as log:
        log.write('Epoch,Loss\n')  # 로그 파일 헤더 작성

        # 학습 루프
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            running_loss = 0.0

            # tqdm을 이용한 진행률 표시
            for mixture_path, bass_path in tqdm(audio_pairs, desc="Training"):  # drums_path -> bass_path로 변경
                # 오디오 파일 로드 (이후 device로 이동)
                mixture_wav = load_wav_to_tensor(mixture_path, config['sampling_rate']).to(device)
                bass_wav = load_wav_to_tensor(bass_path, config['sampling_rate']).to(device)  # drums_wav -> bass_wav로 변경

                # Mel 스펙트로그램 생성
                mixture_mel = mel_spectrogram(mixture_wav,
                                              config['n_fft'],
                                              config['num_mels'],
                                              config['sampling_rate'],
                                              config['hop_size'],
                                              config['win_size'],
                                              config['fmin'],
                                              config['fmax']).to(device)

                bass_mel = mel_spectrogram(bass_wav,  # drums_mel -> bass_mel로 변경
                                           config['n_fft'],
                                           config['num_mels'],
                                           config['sampling_rate'],
                                           config['hop_size'],
                                           config['win_size'],
                                           config['fmin'],
                                           config['fmax']).to(device)

                # 입력 데이터 차원 추가 (batch_size, channels, height, width)
                mixture_mel = mixture_mel.unsqueeze(0)
                bass_mel = bass_mel.unsqueeze(0)

                # 모델 예측
                predicted_bass_mel = model(mixture_mel)  # predicted_drums_mel -> predicted_bass_mel로 변경

                # 손실 계산
                loss = criterion(predicted_bass_mel, bass_mel)  # predicted_drums_mel -> predicted_bass_mel로 변경

                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 손실 기록
                running_loss += loss.item()

            # 에폭별 손실 출력
            epoch_loss = running_loss / len(audio_pairs)
            print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

            # 손실 정보 로그 파일에 기록
            log.write(f"{epoch+1},{epoch_loss:.4f}\n")

            # 모델 체크포인트 저장
            model_save_path = os.path.join(save_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

# 메인 함수
def main():
    # 설정 파일 경로
    config_path = './HiFi/config_v1.json' 
    
    # 설정 로드
    config = load_config(config_path)
    
    # UNet 모델 정의 및 device로 이동
    model = UNet(in_channels=1, out_channels=1).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = torch.nn.MSELoss()  # 손실 함수로 MSE 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 데이터셋 경로
    data_dir = './data'

    # 학습 실행
    train_model(data_dir, model, optimizer, criterion, config, num_epochs=300)

# main 함수 실행
if __name__ == "__main__":
    main()
