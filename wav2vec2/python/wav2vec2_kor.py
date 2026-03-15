import numpy as np
from rknn.api import RKNN
import argparse
import soundfile as sf
import onnxruntime
import scipy
import json  # <--- [추가] JSON 처리를 위해 필수
import time
import os

CHUNK_LENGTH = 20  # 20 seconds
MAX_N_SAMPLES = CHUNK_LENGTH * 16000

# [수정] 하드코딩된 딕셔너리 삭제하고 빈 딕셔너리로 초기화
id_to_token = {}

def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        print("resample_audio: {} HZ -> {} HZ".format(original_sample_rate, desired_sample_rate))
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate

def ensure_channels(waveform, original_channels, desired_channels=1):
    if original_channels != desired_channels:
        print("convert_channels: {} -> {}".format(original_channels, desired_channels))
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_channels

def init_model(model_path, target=None, device_id=None):
    if model_path.endswith(".rknn"):
        # Create RKNN object
        model = RKNN()

        # Load RKNN model
        print('--> Loading model')
        ret = model.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model \"{}\" failed!'.format(model_path))
            exit(ret)
        print('done')

        # init runtime environment
        print('--> Init runtime environment')
        ret = model.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

    elif model_path.endswith(".onnx"):
        model = onnxruntime.InferenceSession(model_path,  providers=['CPUExecutionProvider'])

    return model

def run_model(model, audio_array):
    if 'rknn' in str(type(model)):
        outputs  = model.inference(inputs=audio_array)[0]
    elif 'onnx' in str(type(model)):
        outputs  = model.run(None, {model.get_inputs()[0].name: audio_array})[0]

    return outputs

def release_model(model):
    if 'rknn' in str(type(model)):
        model.release()
    elif 'onnx' in str(type(model)):
        del model
    model = None

def pre_process(audio_array, max_length, pad_value=0):
    array_length = len(audio_array)

    if array_length < max_length:
        pad_length = max_length - array_length
        audio_array = np.pad(audio_array, (0, pad_length), mode='constant', constant_values=pad_value)
    elif array_length > max_length:
        audio_array = audio_array[:max_length]

    return audio_array

# [추가] Vocab.json 로드 함수
def load_vocab(vocab_path):
    global id_to_token
    print(f"--> Loading vocab file: {vocab_path}")
    
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        
        # vocab.json은 보통 {"가": 5, "나": 6} 형태임.
        # 이를 {"5": "가", "6": "나"} 형태로 뒤집어야 함.
        id_to_token = {int(v): k for k, v in vocab_json.items()}
        print(f"--> Vocab loaded successfully! Total tokens: {len(id_to_token)}")
        
    except Exception as e:
        print(f"ERROR: Failed to load vocab file. {e}")
        exit(1)

def compress_sequence(sequence):
    compressed_sequence = [sequence[0]]

    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            compressed_sequence.append(sequence[i])

    return compressed_sequence

def decode(token_ids):
    token_ids = compress_sequence(token_ids)
    transcriptions = []

    for token_id in token_ids:
        # [수정] 로드된 vocab에서 글자 찾기 (없으면 무시)
        token = id_to_token.get(token_id, None)
        
        if token is None:
            continue

        # [수정] 특수 토큰 처리 (Wav2Vec2 표준)
        # 한국어 모델마다 특수 토큰이 조금씩 다를 수 있으나 보통 아래와 같음
        if token in ["<pad>", "<s>", "</s>", "<unk>"]:
            continue
        
        # [수정] 공백 문자 처리 (파이프 '|'를 공백으로 변환)
        if token == "|":
            transcriptions.append(' ')
        else:
            transcriptions.append(token)

    transcription = "".join(transcriptions)
    return transcription

def post_process(output):
    predicted_ids = np.argmax(output, axis=-1)
    transcription = decode(predicted_ids[0])
    return transcription

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wav2vec2 Python Demo', add_help=True)
    # basic params
    parser.add_argument('--model_path', type=str, required=True, help='model path, could be .rknn file or .onnx')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    # [추가] Vocab 파일 경로 입력받기
    parser.add_argument('--vocab_path', type=str, default='../json/vocab.json', help='Path to vocab.json file')
    parser.add_argument('--input', default='../input/wav2vec2_stt_testset') 

    args = parser.parse_args()

    # [추가] Vocab 로드 실행
    load_vocab(args.vocab_path)

    # Set inputs
    # [참고] 테스트할 wav 파일 경로는 필요에 따라 수정하세요
    audio_path = args.input 
    audio_input = []
    print(f"--> Reading audio: {audio_path}")
    for fname in os.listdir(audio_path): 
        if ".wav" in fname:
            try:
                audio_data, sample_rate = sf.read(audio_path+'/'+fname)
            except Exception as e:
                print(f"Error reading audio file: {e}")
                exit(1)
            channels = audio_data.ndim
            audio_data, channels = ensure_channels(audio_data, channels)
            audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
            audio_array = np.array(audio_data, dtype=np.float32)
            audio_array = pre_process(audio_array, MAX_N_SAMPLES)
            audio_array = np.expand_dims(audio_array, axis=0)
            audio_input.append(audio_array)
        else:
            continue

    # Init model
    init_start_time = time.time()
    model = init_model(args.model_path, args.target, args.device_id)
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    print(f'init time: {init_time:.2f}s')
    # Run model
    for audio in audio_input:
        if ".wav" in fname:
            inf_start_time = time.time()
            print("--> Running inference")
            print(audio_path+'/'+fname)
            outputs = run_model(model, audio)
            inf_end_time = time.time()
            inf_time = inf_end_time - inf_start_time

            # Post process
            print("--> Decoding output")
            transcription = post_process(outputs)
            print('\nWav2vec2 output:', transcription)
            print(f'inf time: {inf_time:.2f}s')
        else:
            continue


    # Release model
    release_model(model)
