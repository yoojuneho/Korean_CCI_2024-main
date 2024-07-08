import json

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX = -100  # 무시할 인덱스 값 설정
        self.inp = []  # 입력 데이터를 저장할 리스트
        self.trg = []  # 타겟 데이터를 저장할 리스트
        self.label = []  # 레이블 데이터를 저장할 리스트

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        with open(fname, "r") as f:
            data = json.load(f)  # JSON 파일을 읽어옴

        # 대화 내용을 형식에 맞게 변환하는 함수
        def make_chat(inp):
            chat = ["[Conversation]"] # 리스트를 초기화하고 [Conversation] 태그를 추가하여 대화 내용이 시작됨을 나타냄
            for cvt in inp['conversation']: # 입력 데이터에서 `conversation` 키에 저장된 대화 내용을 순회
                speaker = cvt['speaker'] # 각 대화의 화자 추출
                utterance = cvt['utterance'] # 각 대화의 발언 추출
                chat.append(f"화자{speaker}: {utterance}") # 대화 내용을 화자{speaker}: {utterance} 형식으로 변환하여 chat 리스트에 추가
            chat = "\n".join(chat) # chat 리스트의 모든 요소를 줄바꿈(\n)을 사용하여 하나의 문자열로 결합

            question = f"[Question]\n위 대화의 {inp['category']}" # 질문 텍스트를 `[Question]` 태그와 함께 생성 후 `inp['category']` 값에 따라 질문의 카테고리를 추가
            if (ord(inp['category'][-1]) - ord("가")) % 28 > 0: # 한국어의 조사를 올바르게 추가하기 위해 마지막 글자의 유니코드 값을 확인
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"
                
            chat = chat + "\n\n" + question + "\n\n[Option]\n" # `chat` 문자열에 질문을 추가, `[option]` 태그를 추가하여 선택지 섹션을 시작
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])  # 대화 내용을 형식에 맞게 변환
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
    
            # 토크나이저를 사용하여 입력 데이터 변환
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True, # 생성 프롬프트를 추가함을 의미
                return_tensors="pt", # 결과를 PyTorch 텐서 형식으로 반환
            )

            # 타겟 데이터 설정
            target = ""
            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}" # eos_token: "end of sequence"의 약자로, 시퀀스(문장, 문단 등)의 끝을 나타내는 토큰
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
                
            target = tokenizer(target, # `tokenizer`를 사용하여 타겟 문자열을 토큰화
                        return_attention_mask=False, # 어텐션 마스크를 생성하지 않음
                        add_special_tokens=False, # 특별 토큰을 추가하지 않음
                        return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            # 입력 데이터와 타겟 데이터를 연결하여 최종 입력 및 레이블 생성
            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)  # 데이터셋의 길이 반환

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]  # 주어진 인덱스의 데이터 반환

class DataCollatorForSupervisedDataset(object): # 데이터 배치를 준비하는 역할
    def __init__(self, tokenizer): # 클래스가 초기화 될 때 호출
        self.tokenizer = tokenizer # `tokenizer`는 토크나이저 객체로, 이를 통해 패딩 토큰 값을 얻음

    def __call__(self, instances): # 인스턴스들을 패딩하고, 모델에 필요한 형식으로 변환
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        # 입력 데이터 패딩
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # 레이블 데이터 패딩
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),  # 패딩 토큰이 아닌 부분을 표시하는 attention mask 생성
        )