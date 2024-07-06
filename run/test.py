import argparse # 커맨드 라인 인자 처리를 위한 라이브러리
import json # JSON 파일 입출력을 위한 라이브러리
import tqdm # 진행 상황을 시각적으로 표시하기 위한 라이브러리

import torch # PyTorch 라이브러리
import numpy # 수치 계산을 위한 라이브러리
from transformers import AutoTokenizer, AutoModelForCausalLM # Huggingface의 transformers 라이브러리로 모델과 토크나이저를 불러오기 위함

from src.data import CustomDataset # 사용자 정의 데이터셋 클래스 (위치: `src/data.py`)


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename") # --output: 출력 파일 이름
g.add_argument("--model_id", type=str, required=True, help="huggingface model id") # --model_id: 모델 ID
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer") # --tokenizer: 토크나이저 (선택사항, 기본값은 `model_id`)
g.add_argument("--device", type=str, required=True, help="device to load the model") # --device: 모델을 로드할 디바이스 (예: "cpu", "cuda")
# fmt: on


def main(args):
    # 사전 학습된 모델 로드 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval() # 모델을 평가 모드로 설정

    if args.tokenizer == None: # 토크나이저를 모델 ID로부터 불러옴
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token # `pad_token`을 `eos_token`으로 설정
    
    # 사용자 정의 데이터셋 로드
    dataset = CustomDataset("resource/data/대화맥락추론_test.json", tokenizer) # `CustomDataset`을 사용하여 데이터셋을 로드

    # 답변 매핑 딕셔너리
    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    # 입력 데이터 로드
    with open("resource/data/대화맥락추론_test.json", "r") as f: # 테스트 데이터 JSON 파일을 읽어옴
        result = json.load(f)

    # 데이터셋의 각 항목에 대해 추론 수행
    for idx in tqdm.tqdm(range(len(dataset))):
        inp, _ = dataset[idx] # 현재 인덱스의 데이터를 가져옴
        outputs = model(
            inp.to(args.device).unsqueeze(0) # 입력 데이터를 디바이스로 이동시키고 배치 차원 추가
        )
        logits = outputs.logits[:,-1].flatten() # 마지막 토근에 대한 로짓을 추출하고 평탄화
        
        # 선택지 `A`, `B`, `C`에 대한 확률 계산
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer.vocab['A']], # `A` 선택직의 로짓 값
                        logits[tokenizer.vocab['B']], # `B` 선택직의 로짓 값
                        logits[tokenizer.vocab['C']], # `C` 선택직의 로짓 값
                    ]
                ),
                dim=0, # 소프트맥스 함수의 차원 설정
            )
            .detach() # 그래프에서 분리
            .cpu() # CPU 메모리로 이동
            .to(torch.float32) # float32로 형변환
            .numpy() # 넘파이 배열로 변환
        )

        # 가장 높은 확률을 가진 답변 선택
        result[idx]["output"] = answer_dict[numpy.argmax(probs)]

    # 결과를 JSON 파일로 저장
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

# 스크립트 실행
if __name__ == "__main__":
    exit(main(parser.parse_args())) # 커맨드 라인 인자를 파싱하여 메인 함수를 실행