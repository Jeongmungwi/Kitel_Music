from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

# x로부터 mel spectrogram을 얻음 
def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

# checkpoint에 원하는 디렉토리가 있는지 체크
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(a):
    generator = Generator(h).to(device) # 모델 초기화 # h로 뭘 받는건가?? 잘 모르겠네...

    state_dict_g = load_checkpoint(a.checkpoint_file, device)  # chekpoint 불러오기
    generator.load_state_dict(state_dict_g['generator']) # 현재 딕셔너리 상태 불러오기

    filelist = os.listdir(a.input_wavs_dir) # input 파일 긁어오기

    os.makedirs(a.output_dir, exist_ok=True) # 출력 디렉토리 생성하기

    generator.eval() # 모델 평가 모드로 설정
    generator.remove_weight_norm() # ㅏ중치 제거
    # 파일 리스트 안에 있는 파일들을 정규화하고, 멜 스펙트로그램을 얻고, 생성기로 오디오 생성 
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='./test')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', default='./HiFi/generator_v1')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config_v1.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
