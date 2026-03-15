import os
import numpy as np
import soundfile as sf
import glob

# ==========================================
# 설정 (5초 모델 기준)
# ==========================================
TARGET_SR = 16000
CHUNK_SECONDS = 5
TARGET_LEN = CHUNK_SECONDS * TARGET_SR  # 80,000
INPUT_DIR = "../input"       # 여기에 wav 파일들을 넣어두세요
OUTPUT_DIR = "../calibration_data" # 변환된 npy가 저장될 곳
DATASET_FILE = "../dataset.txt"    # rknn에 먹여줄 목록 파일

def prepare_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    if not wav_files:
        print(f"❌ Error: '{INPUT_DIR}' 폴더에 .wav 파일이 없습니다!")
        return

    print(f"--> 발견된 WAV 파일: {len(wav_files)}개")
    npy_paths = []

    for wav_path in wav_files:
        # 1. 로드
        audio, sr = sf.read(wav_path)
        
        # 2. 모노 변환
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # 3. 샘플레이트 체크 (리샘플링은 생략, 16k 가정)
        if sr != TARGET_SR:
            print(f"⚠️ Warning: {wav_path} is {sr}Hz. (Expected {TARGET_SR}Hz)")

        # 4. 5초(80000) 길이 맞추기 (자르기 or 패딩)
        if len(audio) > TARGET_LEN:
            audio = audio[:TARGET_LEN]
        else:
            padding = np.zeros(TARGET_LEN - len(audio))
            audio = np.concatenate((audio, padding))
            
        # 5. 형변환 및 배치 차원 추가 (1, 80000)
        # 중요: float32로 저장해야 RKNN이 읽어서 양자화 계산함
        data = audio.astype(np.float32).reshape(1, -1)
        
        # 6. 저장
        base_name = os.path.basename(wav_path).replace(".wav", ".npy")
        save_path = os.path.join(OUTPUT_DIR, base_name)
        np.save(save_path, data)
        
        # 절대 경로로 저장하는 게 안전함
        npy_paths.append(os.path.abspath(save_path))
        print(f"  Converted: {base_name} -> Shape {data.shape}")

    # 7. dataset.txt 생성
    with open(DATASET_FILE, 'w') as f:
        for path in npy_paths:
            f.write(path + '\n')
            
    print(f"\n✅ 완료! '{DATASET_FILE}' 생성됨.")
    print(f"이제 convert.py 실행할 때 dataset='./dataset.txt' 하면 됩니다.")

if __name__ == "__main__":
    prepare_dataset()
