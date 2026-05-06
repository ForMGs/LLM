"""
목적
    - train.txt에 있는 짧은 문장 데이터를 읽는다.
    - GPT-2 토크나이저로 숫자로 바꾼다.
    - 작은 GPT-2 구조 모델을 새로 만든다.
    - 그 데이터를 보고 학습시킨다.
    - tinyLLM 폴더에 저장한다.
    즉 train.py는 작은 언어모델을 학습시키는 코드. 저장소의 train.py는 Hugging Face의 transformers
    와 datasets를 사용하고, GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, Trainer,
    TrainingArguments 를 가져와 학습을 구성한다.
"""


# transformers 에서 GPT2Config , GPT2LMHeadModel, GPT2TokenizerFast, Trainer,
# TrainingArguments 를 가져오고, datasets 의 Dataset을 사용.
from transformers import(
    GPT2Config,         #GPT-2 모델 설계도 설정 클래스
    GPT2LMHeadModel,    #GPT-2 구조의 언어모델 클래스
    GPT2TokenizerFast,  #문장을 토큰 숫자로 바꿔주는 토크나이저
    Trainer,            #학습 루프를 대신 돌려주는 도구
    TrainingArguments   #학습 옵션을 저장하는 설정 클래스
)   
from datasets import Dataset    #학습 데이터를 Hugging Face 학습용  데이터셋 형태로 바꾸는 클래스.


#GPT-2 토크나이저 사용
# ------------------------
# Tokenizer
# ------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Use default GPT-2 EOS and set PAD to EOS for causal LM training
# 문장끝을 채울 때는 문장 끝 토큰을 대신 쓴다.( 학습할 시 길이가 틀리기에 padding을 채우기위함)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Build regular prompt -> completion dataset (no roles)
# prompt     : 사용자 입력, 문제
# completion : 모델이 맞춰야 하는 정답
# -------------------------
def build_example(prompt, completion):

    text = f"{prompt}\n{completion}{tokenizer.eos_token}"

    # 텍스트를 숫자로 변경. (truncation=True 는 너무 길면 자른다.)
    tokens = tokenizer(text, truncation=True)
    input_ids = tokens["input_ids"]

    # input_ids 사이즈 만큼 label에 -100으로 채워 생성. (label 은 정답지 딥러닝에서 -100은 이 위치는 loss 계산하지마라는 의미)
    labels = [-100] * len(input_ids)

    #어디까지가 prompt인지 계산하는 영역.
    prompt_prefix = f"{prompt}\n"
    prefix_ids = tokenizer(prompt_prefix, truncation=True)["input_ids"]
    start = len(prefix_ids)

    #학습용만 넣음. prompt는 -100 
    labels[start:] = input_ids[start:]

    return {
        "input_ids": input_ids,                         #모델에게 넣을 토큰 숫자
        "attention_mask": tokens["attention_mask"],     #어떤 토큰을 실제 입력으로 볼지 표시
        "labels": labels,                               #모델이 맞춰야 할 정답.
    }

# 파일을 읽는 함수.
def load_examples_from_file(path):

    #파일 열기.
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n")]
    examples = []

    for block in blocks:
        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]

        if len(lines) < 2:
            continue

        prompt = lines[0]
        completion = lines[1]

        examples.append(build_example(prompt, completion))

    return examples

examples = load_examples_from_file("train.txt")
dataset = Dataset.from_list(examples)

"""
Model 설정
"""
config = GPT2Config(
    vacab_size=len(tokenizer),
    n_positions=256,
    n_ctx=256,
    n_embd=384,
    n_layer=6,
    n_head=6,
)
#GPT-2 구조는 사용하지만, 모델의 지식은 랜덤 상태에서 시작한다.
model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))

"""
Training : 학습 설정 만들기
"""
args = TrainingArguments(
    output_dir="tinyLLM",           #학습 결과를 tinyLLM 폴더에 저장하겠다.
    # overwrite_output_dir=True,      #tinyLLM 폴더가 이미 있어도 덮어써라
    per_device_train_batch_size=2,  #한번에 학습할 예제 개수 = 2
    num_train_epochs=200,           #전체 데이터를 200번 반복해서 학습하겠다.
    learning_rate=5e-4,             #학습률 (5e-4 = 0.0005) 모델이 틀렸을 때 얼마나 크게 수정할지
    logging_steps=10,               #10번 step 마다 로그를 찍어라
    save_steps=500,                 #500 step마다 중간 저장해라
    save_total_limit=1,             #저장된 checkpoint는 최대 1개만 유지해라
    fp16=False,                     #16-bit 반정밀도 학습을 쓰지 않겠다.
    report_to="none",               #wandb 같은 외부 실험 추적 도구에 보고하지 않겠다.
    dataloader_num_workers=0,       #데이터 로딩 작업자를 추가로 만들지 않겠다.

)

#Trainer 는 학습을 대신 돌려주는 도구.
trainer =Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("tinyLLM")
tokenizer.save_pretrained("tinyLLM")