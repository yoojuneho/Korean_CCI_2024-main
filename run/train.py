import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from src.data import CustomDataset, DataCollatorForSupervisedDataset

# argparse를 사용하여 커맨드 라인 인자 처리
parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")  # 모델 파일 경로
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")  # 토크나이저 경로 (선택사항)
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")  # 모델 저장 경로
g.add_argument("--batch_size", type=int, default=1, help="batch size (both train and eval)")  # 배치 사이즈
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")  # 기울기 누적 스텝 수
g.add_argument("--warmup_steps", type=int, help="scheduler warmup steps")  # 스케줄러 웜업 스텝 수
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")  # 학습률
g.add_argument("--epoch", type=int, default=5, help="training epoch")  # 학습 에포크 수

def main(args):
    # 사전 학습된 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # 토크나이저 로드 (선택사항)
    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token  # pad_token을 eos_token으로 설정

    # 사용자 정의 데이터셋 로드
    train_dataset = CustomDataset("resource/data/대화맥락추론_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/대화맥락추론_dev.json", tokenizer)

    # 데이터셋을 Huggingface Datasets 객체로 변환
    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)  # 데이터 콜레이터 설정

    # SFT(Supervised Fine-tuning Trainer)Config(구성)를 사용하여 학습 설정
    training_args = SFTConfig( # SFTConfig 클래스를 사용하여 학습과정 설정 정의
        output_dir=args.save_dir,  # 모델 저장 디렉토리
        overwrite_output_dir=True,  # 기존 디렉토리 존재할 경우 덮어쓰기
        do_train=True,  # 학습 수행
        do_eval=True,  # 평가 수행
        eval_strategy="epoch",  # 평가 전략, `epoch`로 설정했기 때문에 각 에포크가 끝날 때마다 평가
        per_device_train_batch_size=args.batch_size,  # 학습 배치 사이즈
        per_device_eval_batch_size=args.batch_size,  # 평가 배치 사이즈
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 기울기 누적 스텝 수
        learning_rate=args.lr,  # 학습률 설정
        weight_decay=0.1,  # 가중치 감소율 설정
        num_train_epochs=args.epoch,  # 학습 에포크 수(전체 합습 데이터셋을 몇 번 반복할지 설정)
        max_steps=-1,  # 최대 스텝 수 (제한 없음)
        lr_scheduler_type="cosine",  # 학습률 스케줄러 타입
        warmup_steps=args.warmup_steps,  # 웜업(학습 초기에 학습률을 서서히 증가) 스텝 수
        log_level="info",  # 로그 레벨, `info`로 설정하면 일반 정보를 기록
        logging_steps=1,  # 로그 기록 스텝 수(몇 스텝마다 로그를 기록할지)
        save_strategy="epoch",  # 모델 저장 전략, `epoch`로 설정하면 각 에포크가 끝날 때마다 모델 저장
        save_total_limit=5,  # 저장할 모델 최대 개수, 이 개수를 초과하면 가장 오래된 모델부터 삭제
        bf16=True,  # bf16 사용 여부
        gradient_checkpointing=True,  # 기울기 체크포인트 사용 여부
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 기울기 체크포인트 옵션
        max_seq_length=1024,  # 최대 시퀀스 길이
        packing=True,  # 패킹 사용 여부
        seed=42,  # 시드 값
    )

    # SFTTrainer 클래스를 사용하여 트레이너 객체를 설정
    trainer = SFTTrainer(
        model=model, # 사전 학습된 언어 모델
        tokenizer=tokenizer, # 사전 학습된 토크나이저
        train_dataset=train_dataset, # 학습에 사용할 데이터셋
        eval_dataset=valid_dataset, # 평가에 사용할 데이터셋
        data_collator=data_collator, # 배치 내에서 샘플들을 병합하고, 필요 시 패딩을 추가하는 역할을 하는 데이터 콜레이터 객체
        args=training_args, # 모델 학습에 필요한 설정들을 포함하는 `SFTConfig` 객체
    ) 

    # 모델 학습
    trainer.train() # `train()` 메서드는 `SFTTrainer` 클래스의 메서드

if __name__ == "__main__":
    exit(main(parser.parse_args()))  # 커맨드 라인 인자를 파싱하여 메인 함수 실행