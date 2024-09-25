import wave
import contextlib

def get_wav_sampling_rate(file_path):
    with contextlib.closing(wave.open(file_path, 'rb')) as wav_file:
        sample_rate = wav_file.getframerate()
        return sample_rate

# 샘플 사용 예시
file_path = './LJ_test/LJ001-0001.wav'
sampling_rate = get_wav_sampling_rate(file_path)
print(f"Sampling rate: {sampling_rate} Hz")

file_path_2 = './musdb18hq_mono/A Classic Education - NightOwl/bass.wav'
sampling_rate_2 = get_wav_sampling_rate(file_path_2)
print(f"Sampling rate_2: {sampling_rate_2} Hz")