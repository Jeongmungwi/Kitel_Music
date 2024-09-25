import numpy as np
import matplotlib.pyplot as plt

# 멜 스펙트로그램을 플롯하는 함수
def plot_mel_spectrogram(npy_file_path):
    # .npy 파일 로드
    mel_spectrogram = np.load(npy_file_path)

    # Mel Spectrogram Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency Bins')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# 실행 예시
npy_file_path = './npy_test/bass_300.npy'  # .npy 파일 경로 (저장된 멜 스펙트로그램 파일 경로)
# npy_file_path = './npy_test/bass_14.npy'
plot_mel_spectrogram(npy_file_path)
