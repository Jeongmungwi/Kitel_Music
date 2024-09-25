import torch
import torchaudio
import torch.nn.functional as F
import json
import os
import sys
from block import UNet, UNet1D  # B 구성에서 사용하는 mel-spectrogram U-Net

sys.path.append("./HiFi")
from HiFi.meldataset import mel_spectrogram, MAX_WAV_VALUE
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

# 텐서 크기 맞추기 (패딩 추가)
def match_tensor_lengths(tensor_a, tensor_b):
    length_a = tensor_a.size(-1)
    length_b = tensor_b.size(-1)

    if length_a < length_b:
        padding = length_b - length_a
        tensor_a = F.pad(tensor_a, (0, padding))
    elif length_b < length_a:
        padding = length_a - length_b
        tensor_b = F.pad(tensor_b, (0, padding))

    return tensor_a, tensor_b

# 테스트 함수
def test_model(test_dir, model_a, model_b, config, generator):
    # 테스트 데이터 경로에서 mixture 파일 로드
    for mixture_file in os.listdir(test_dir):
        if mixture_file.startswith("mixture_") and mixture_file.endswith(".wav"):
            mixture_path = os.path.join(test_dir, mixture_file)
            n = mixture_file.split("_")[1].split(".")[0]  # mixture_n.wav에서 n 추출
            mixture_wav = load_wav_to_tensor(mixture_path, config['sampling_rate']).to(device)

            # A 구성의 예측 결과
            predicted_bass_wav_a = forward_a(mixture_wav, model_a)

            # B 구성의 예측 결과
            predicted_bass_wav_b = forward_b(mixture_wav, model_b, config, generator)

            # 두 텐서의 길이를 맞추기 위한 패딩 처리
            max_length = max(predicted_bass_wav_a.size(-1), predicted_bass_wav_b.size(-1))

            # predicted_bass_wav_a에 패딩 추가
            if predicted_bass_wav_a.size(-1) < max_length:
                padding = max_length - predicted_bass_wav_a.size(-1)
                predicted_bass_wav_a = F.pad(predicted_bass_wav_a, (0, padding))

            # predicted_bass_wav_b에 패딩 추가
            if predicted_bass_wav_b.size(-1) < max_length:
                padding = max_length - predicted_bass_wav_b.size(-1)
                predicted_bass_wav_b = F.pad(predicted_bass_wav_b, (0, padding))

            # A와 B의 결과를 결합 (평균값)
            combined_predicted_bass_wav = (predicted_bass_wav_a + predicted_bass_wav_b) / 2

            # mix_result 폴더가 없으면 생성
            output_dir = "./mix_result"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 결합된 예측 결과 저장
            output_path = os.path.join(output_dir, f"bass_{n}_predicted.wav")

            # 3D 텐서를 2D로 변환 (channels, samples)
            if combined_predicted_bass_wav.dim() == 1:
                combined_predicted_bass_wav = combined_predicted_bass_wav.unsqueeze(0)

            # 텐서에서 detach() 후 저장
            torchaudio.save(output_path, combined_predicted_bass_wav.detach().cpu(), config['sampling_rate'])
            print(f"결과 저장: {output_path}")

# 테스트 메인 함수
def main():
    # 설정 파일 로드
    config_path = './HiFi/config_v1.json'
    config = load_config(config_path)

    # HiFi-GAN Generator 모델 정의 및 체크포인트 로드
    generator = Generator(AttrDict(config)).to(device)
    checkpoint_file = './HiFi/generator_v1'
    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # A 구성의 1D U-Net 모델 정의 및 학습된 체크포인트 로드
    model_a = UNet1D(in_channels=1, out_channels=1).to(device)
    model_a.load_state_dict(torch.load('./ONet_Checkpoint_bass/model_a_epoch_10.pth', map_location=device))

    # B 구성의 이미 학습된 mel-spectrogram U-Net 모델 로드
    model_b = UNet(in_channels=1, out_channels=1).to(device)
    model_b.load_state_dict(torch.load('./bass_ck_0924/unet_epoch_300.pth', map_location=device))

    # 테스트 데이터 경로
    test_dir = './mix_test'

    # 테스트 실행
    test_model(test_dir, model_a, model_b, config, generator)

if __name__ == "__main__":
    main()