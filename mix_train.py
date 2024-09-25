import torch
import torchaudio
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
import sys
from block import UNet, UNet1D  # B 구성에서 사용하는 mel-spectrogram U-Net

sys.path.append("./HiFi")
from HiFi.meldataset import mel_spectrogram
from HiFi.models import Generator  # HiFi-GAN Generator 함수
from HiFi.inference_e2e import load_checkpoint  # HiFi-GAN checkpoint 로드 함수
from HiFi.env import AttrDict

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
            bass_files = sorted([f for f in os.listdir(folder_path) if f.startswith("bass_") and f.endswith(".wav")])
            for mix_file, bass_file in zip(mixture_files, bass_files):
                mixture_path = os.path.join(folder_path, mix_file)
                bass_path = os.path.join(folder_path, bass_file)
                audio_pairs.append((mixture_path, bass_path))
    return audio_pairs

# A 구성: mixture.wav를 1D U-Net에 입력하여 predicted_bass_wav 생성
def forward_a(mixture_wav, model_a):
    mixture_wav = mixture_wav.unsqueeze(0)  # (1, channels, samples)
    predicted_bass_wav = model_a(mixture_wav)
    return predicted_bass_wav.squeeze(0)  # (channels, samples)

# B 구성에서 HiFi-GAN을 사용하여 mel-spectrogram을 waveform으로 변환
def forward_b(mixture_wav, model_b, config, generator):
    # Mel-spectrogram 생성
    mixture_mel = mel_spectrogram(mixture_wav,
                                  config['n_fft'],
                                  config['num_mels'],
                                  config['sampling_rate'],
                                  config['hop_size'],
                                  config['win_size'],
                                  config['fmin'],
                                  config['fmax']).to(device)

    mixture_mel = mixture_mel.unsqueeze(0)  # (1, channels, height, width)

    # B 구성의 U-Net을 통해 mel-spectrogram 예측
    predicted_bass_mel = model_b(mixture_mel)

    # predicted_bass_mel의 차원 변환: [batch_size, 1, n_mels, time_steps] -> [batch_size, n_mels, time_steps]
    predicted_bass_mel = predicted_bass_mel.squeeze(1)  # (batch_size, n_mels, time_steps)

    # HiFi-GAN generator를 사용하여 mel-spectrogram을 waveform으로 변환
    generator.eval()  # Generator 평가 모드로 설정
    
    # weight_norm이 적용된 경우에만 제거
    for l in generator.modules():
        if isinstance(l, torch.nn.Conv1d) or isinstance(l, torch.nn.ConvTranspose1d):
            try:
                torch.nn.utils.remove_weight_norm(l)
            except ValueError:
                pass  # weight_norm이 없는 경우 패스

    with torch.no_grad():
        y_g_hat = generator(predicted_bass_mel)  # mel-spectrogram으로부터 waveform 생성
        predicted_bass_wav = y_g_hat.squeeze()

    return predicted_bass_wav

# 두 구성의 출력을 결합하여 하나의 waveform으로 만들고 손실 계산
def compute_combined_loss(mixture_wav, bass_wav, model_a, model_b, criterion, config, generator):
    # A 구성의 출력
    predicted_bass_wav_a = forward_a(mixture_wav, model_a)

    # B 구성의 출력 (mel-spectrogram 기반 U-Net 및 HiFi-GAN을 사용)
    predicted_bass_wav_b = forward_b(mixture_wav, model_b, config, generator)

    # B 구성 출력에 배치 차원 추가 (1D -> 2D)
    if predicted_bass_wav_b.dim() == 1:
        predicted_bass_wav_b = predicted_bass_wav_b.unsqueeze(0)

    # 최대 길이에 맞춰 크기 맞추기 (패딩 추가)
    max_length = max(predicted_bass_wav_a.size(1), predicted_bass_wav_b.size(1))

    # predicted_bass_wav_a에 패딩 추가
    if predicted_bass_wav_a.size(1) < max_length:
        padding = max_length - predicted_bass_wav_a.size(1)
        predicted_bass_wav_a = F.pad(predicted_bass_wav_a, (0, padding))

    # predicted_bass_wav_b에 패딩 추가
    if predicted_bass_wav_b.size(1) < max_length:
        padding = max_length - predicted_bass_wav_b.size(1)
        predicted_bass_wav_b = F.pad(predicted_bass_wav_b, (0, padding))

    # A와 B 구성의 출력을 결합 (예: 평균)
    combined_predicted_bass_wav = (predicted_bass_wav_a + predicted_bass_wav_b) / 2

    # 실제 bass.wav와 비교하여 손실 계산
    loss = criterion(combined_predicted_bass_wav, bass_wav)

    return loss

# A 구성만을 위한 학습 함수
def train_a_only(data_dir, model_a, model_b, optimizer_a, criterion, config, generator, num_epochs=10):
    # 체크포인트 경로 설정
    checkpoint_dir = './ONet_Checkpoint'
    
    # 체크포인트 폴더가 없으면 생성
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    audio_pairs = get_audio_file_pairs(data_dir)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for mixture_path, bass_path in tqdm(audio_pairs, desc=f"Epoch {epoch+1}/{num_epochs}"):
            mixture_wav = load_wav_to_tensor(mixture_path, config['sampling_rate']).to(device)
            bass_wav = load_wav_to_tensor(bass_path, config['sampling_rate']).to(device)

            # 손실 계산
            loss = compute_combined_loss(mixture_wav, bass_wav, model_a, model_b, criterion, config, generator)

            # A 구성만을 위한 역전파 및 최적화
            optimizer_a.zero_grad()
            loss.backward()
            optimizer_a.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(audio_pairs)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # 모델 저장
        model_save_path = os.path.join(checkpoint_dir, f"model_a_epoch_{epoch+1}.pth")
        torch.save(model_a.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

# 학습 메인 함수
def main():
    # 설정 파일 로드
    config_path = './HiFi/config_v1.json'
    config = load_config(config_path)

    # HiFi-GAN Generator 모델 정의 및 체크포인트 로드
    generator = Generator(AttrDict(config)).to(device)
    checkpoint_file = './HiFi/generator_v1'
    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # A 구성의 1D U-Net 모델 정의 및 초기화
    model_a = UNet1D(in_channels=1, out_channels=1).to(device)

    # B 구성의 이미 학습된 mel-spectrogram U-Net 모델 로드
    model_b = UNet(in_channels=1, out_channels=1).to(device)
    model_b.load_state_dict(torch.load('./bass_ck_0924/unet_epoch_300.pth', map_location=device))
    model_b = model_b.to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = torch.nn.MSELoss()  # MSE 손실 함수 사용
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=config['learning_rate'])  # A 구성만 학습

    # 학습 데이터 경로
    data_dir = './data'

    num_epochs = 10

    # A 구성 학습 실행
    train_a_only(data_dir, model_a, model_b, optimizer_a, criterion, config, generator, num_epochs=num_epochs)

if __name__ == "__main__":
    main()
