import os
import re
import json
import time
import requests
import pandas as pd
import nlptutti as metrics
from tqdm import tqdm
import sherpa_onnx
import wave
import numpy as np

class SherpaOnnxInference:
    def __init__(self, model_dir):
        """
        Sherpa-ONNX 로컬 모델 기반 STT 추론 클래스
        """
        # 모델 파일 경로 설정 (다운로드 받은 폴더 경로)
        encoder = os.path.join(model_dir, "encoder-epoch-99-avg-1.int8.onnx")
        decoder = os.path.join(model_dir, "decoder-epoch-99-avg-1.int8.onnx")
        joiner = os.path.join(model_dir, "joiner-epoch-99-avg-1.int8.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        # Recognizer 설정
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )

    def read_wave(self, wave_filename):
        """WAV 파일을 읽어 sample rate 확인 및 numpy 배열로 변환"""
        with wave.open(wave_filename, "rb") as f:
            assert f.getnchannels() == 1, "모노 파일만 지원합니다."
            assert f.getsampwidth() == 2, "16-bit PCM 포맷이어야 합니다."
            num_samples = f.getnframes()
            samples = f.readframes(num_samples)
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768
            return samples_float32, f.getframerate()

    def stt_sender(self, path):
        """
        로컬 ONNX 모델을 통한 추론 (에러 수정 버전)
        """
        try:
            samples, sample_rate = self.read_wave(path)
            
            # 스트림 생성
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            
            # tail padding 추가 (인식 성능 향상)
            tail_paddings = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
            stream.accept_waveform(sample_rate, tail_paddings)
            stream.input_finished()

            # 디코딩 루프
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

            # [수정 포인트] 최신 버전은 결과가 바로 문자열로 반환될 수 있습니다.
            result = self.recognizer.get_result(stream)
            
            if isinstance(result, str):
                return result.strip()
            else:
                return result.text.strip()
            
        except Exception as e:
            print(f"❌ 추론 오류 발생 ({path}): {e}")
            return ""

def remove_spaces(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", "", text)
    
def natural_key(s):
    return [
        int(t) if t.isdigit() else t
        for t in re.split(r'(\d+)', os.path.basename(s))
    ]
    
def test_csv(test_file, clova_model, save_dir):
    df = pd.read_csv(test_file)
    df = df[["FileName", "gt"]]
    results = []
    
    for idx, row in tqdm(df.iterrows()):
        file_path = row["FileName"]
        answer = remove_spaces(str(row["gt"]))

        # 파일 존재 확인
        if not os.path.exists(file_path):
            print(f"[WARN] 파일 없음: {file_path}")
            results.append([file_path, row["gt"], "", None, None])
            continue

        # 추론
        start_time = time.time()
        inf_text = clova_model.stt_sender(file_path)
        process_time = time.time() - start_time
        inf_no_space = remove_spaces(inf_text)

        # CER 계산
        cer = metrics.get_cer(answer, inf_no_space)["cer"]

        results.append([
            file_path,
            answer,
            inf_no_space,
            cer,
            process_time
        ])
    
    result_df = pd.DataFrame(results, columns=["FileName", "gt", "inf", "cer", "process_time"])
    avg_cer = result_df['cer'].mean()
    avg_time = result_df['process_time'].mean()
    result_df.to_csv(f"{save_dir}/{test_file.split('/')[-1].replace('.csv','')}_result_CER{avg_cer*100:.2f}_TIME{avg_time:.2f}_int8.csv", index=False, encoding="utf-8-sig")
    
    print(f"AVG CER: {avg_cer}")
    print(f"AVG TIME: {avg_time}")
    
def test_dir(test_file, clova_model):
    print("Input is a directory")

    results = []  # (filename, stt_text)
    audio_paths = []

    for root, _, files in os.walk(test_file):
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                audio_paths.append(os.path.join(root, f))

    audio_paths = sorted(audio_paths, key=natural_key)

    for audio_path in tqdm(audio_paths):
        inf_text = clova_model.stt_sender(audio_path)
        results.append((os.path.basename(audio_path), inf_text))

    # TXT 저장 (파일명 \t 텍스트)
    output_txt = "stt_texts_only.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        for _, stt_text in results:
            f.write(f"{stt_text}\n")

    print(f"Saved {len(results)} texts to {output_txt}")
    
def test_file(test_file, clova_model):
    start_time = time.time()
    inf_text = clova_model.stt_sender(test_file)
    process_time = time.time() - start_time 
    print(f"{test_file} processed for {process_time:.3f}")
    print(inf_text)


if __name__ == '__main__':
    model_path = '/nas04/nlp_sk/STT/sherpa-onnx-streaming-zipformer-korean-2024-06-16'
    stt_model = SherpaOnnxInference(model_path)
    save_dir = './eval_result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    test_csv_list = [
        # '/nas04/nlp_sk/STT/data/test/007.저음질_eval_p.csv',
        # '/nas04/nlp_sk/STT/data/test/009.한국어_강의_eval_p.csv',
        # '/nas04/nlp_sk/STT/data/test/010.회의음성_eval_p.csv',
        # '/nas04/nlp_sk/STT/data/test/012.상담음성_eval_p.csv',
        '/nas04/nlp_sk/STT/data/test/eval_clean_p.csv',
        '/nas04/nlp_sk/STT/data/test/eval_other_p.csv',
        # '/nas04/nlp_sk/STT/data/test/modelhouse_2m_noheater.csv',
        # '/nas04/nlp_sk/STT/data/test/modelhouse_2m.csv',
        # '/nas04/nlp_sk/STT/data/test/modelhouse_3m.csv',
        # '/nas04/nlp_sk/STT/data/test/7F_HJY.csv',
        # '/nas04/nlp_sk/STT/data/test/7F_KSK.csv'
    ]
    
    test_file_path = '/nas04/nlp_sk/STT/sherpa-onnx-streaming-zipformer-korean-2024-06-16/test_wavs/0.wav'
    test_file(test_file_path, stt_model)
    
    # for test_file in test_csv_list:
    #     print(f"[Start Processing] {test_file}")
    #     test_csv(test_file, stt_model, save_dir)
    #     print(f"[Test Done] {test_file}")
        