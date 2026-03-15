import onnx
from onnxsim import simplify
import argparse
import os

def resize_onnx_model(input_model_path, output_model_path, chunk_length_sec):
    # 1. 설정
    sample_rate = 16000
    new_length = chunk_length_sec * sample_rate # 예: 20초 * 16000 = 320,000
    
    print(f"--> Loading model: {input_model_path}")
    try:
        model = onnx.load(input_model_path)
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다: {input_model_path}")
        return

    # 2. 입력 노드 이름 자동 찾기 (중요!)
    # 모델마다 입력 이름이 'input', 'input_values', 'speech' 등으로 다를 수 있음
    input_node_name = model.graph.input[0].name
    print(f"--> Detected input node name: '{input_node_name}'")

    # 3. 변경할 쉐이프 정의 (배치사이즈 1 고정)
    # { "input_node_name": [1, 320000] }
    input_shapes = {input_node_name: [1, new_length]}
    print(f"--> Target Input Shape: {input_shapes}")

    # 4. onnxsim을 사용하여 쉐이프 덮어쓰기 및 내부 구조 단순화 (Simlify)
    # 이 과정에서 dynamic shape이 static(고정)으로 바뀌고, 내부 연산이 정해진 길이에 맞춰 최적화됨.
    print("--> Applying onnx-simplifier (This might take a while)...")
    try:
        model_simp, check = simplify(model, overwrite_input_shapes=input_shapes)
    except Exception as e:
        print(f"Error during simplification: {e}")
        print("힌트: onnxsim 버전 문제일 수 있습니다. (추천: pip install onnxsim==0.4.8)")
        return

    # 5. 검증 및 저장
    if check:
        print("--> Check success! Saving model...")
        onnx.save(model_simp, output_model_path)
        print(f"✅ Success! Saved to: {output_model_path}")
    else:
        print("--> Check failed! Saving anyway but model might be broken...")
        onnx.save(model_simp, output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize existing ONNX model input length")
    
    # 입력 파일 경로 (기본값 설정)
    parser.add_argument("--input", type=str, default="./wav2vec-xls-r-300m.onnx",
                        help="Path to the original input ONNX file")
    
    # 원하는 시간 (초 단위)
    parser.add_argument("--chunk_length", type=int, required=True, default=20,
                        help="Target audio length in seconds (e.g., 5 or 20)")
    
    args = parser.parse_args()

    # 저장할 파일명 자동 생성 (예: wav2vec-xls-r-300m_20s.onnx)
    base_name = os.path.splitext(args.input)[0]
    output_path = f"{base_name}_{args.chunk_length}s.onnx"

    resize_onnx_model(args.input, output_path, args.chunk_length)
