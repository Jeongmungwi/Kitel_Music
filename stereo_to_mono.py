import os
from pydub import AudioSegment
import shutil
from tqdm import tqdm

# 원본 폴더와 타겟 폴더 경로 설정
source_folder = './musdb18hq/test'
target_folder = './data_test'

# 새로운 폴더가 없다면 생성
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 총 파일 수 계산
total_files = sum([len(files) for _, _, files in os.walk(source_folder) if any(file.endswith('.wav') for file in files)])

# 각 하위 폴더를 순회하며 파일 변환
with tqdm(total=total_files, desc="Converting and splitting files") as pbar:
    for subdir, _, files in os.walk(source_folder):
        # 타겟 디렉토리 구조 유지
        relative_path = os.path.relpath(subdir, source_folder)
        target_subdir = os.path.join(target_folder, relative_path)

        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)

        # 각 .wav 파일을 모노로 변환하고 샘플링 레이트 변경 및 30초 단위로 나누기
        for file in files:
            if file.endswith('.wav'):
                source_file_path = os.path.join(subdir, file)
                audio = AudioSegment.from_wav(source_file_path)
                mono_audio = audio.set_channels(1).set_frame_rate(22050)

                # 파일을 30초 단위로 분할
                segment_duration_ms = 10 * 1000  # 30초를 밀리초로 변환
                audio_length_ms = len(mono_audio)
                num_segments = (audio_length_ms // segment_duration_ms) + 1

                for i in range(num_segments):
                    start_time = i * segment_duration_ms
                    end_time = min((i + 1) * segment_duration_ms, audio_length_ms)
                    segment = mono_audio[start_time:end_time]

                    # 분할된 파일 이름 생성 (예: 'A Classic Education - NightOwl_1.wav')
                    file_name, file_ext = os.path.splitext(file)
                    segment_file_name = f"{file_name}_{i+1}{file_ext}"
                    target_file_path = os.path.join(target_subdir, segment_file_name)

                    # 분할된 파일 저장
                    segment.export(target_file_path, format='wav')

                # 진행률 업데이트
                pbar.update(1)

print("모든 파일이 성공적으로 변환 및 분할되었습니다.")
