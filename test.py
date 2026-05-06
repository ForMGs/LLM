"""
학습된 모델을 실제로 불러와서 "입력을 넣으면 어떤 문장을 생성하는 지 " 확인 하는 파일

transformers에서 pipeline 가져오기
                ↓
tinyLLM 폴더에서 모델과 토크나이저 불러오기
                ↓
테스트용 prompt 만들기
                ↓
모델에게 이어서 10토큰 생성시키기
                ↓
생성된 전체 문장 출력하기 
"""

from transformers import pipeline

pipe = pipeline(
    "text-generation",          #파이프라인 작업종류 (텍스트 생성 작업을 하겠다.)
    model="./tinyLLM",          #불러올 모델 위치
    tokenizer="./tinyLLM",      #tinyLLM 폴더에서 불러오겠다.
)

prompt = "User: 하이\nAssistant:"

out = pipe(prompt, max_new_tokens=10, do_sample=False)

print(out[0]["generated_text"])