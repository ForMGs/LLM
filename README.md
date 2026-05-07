# How to be AI Engineer..?
## 1. How to prompt AI...?
### Prompt Engineering


## 2. How to integrate database with AI. (RAG)
### RAG ? AI가 모르는 내 데이터/문서를 검색해서 답변하게 만드는 구조.
#### - Embedding 개념
  - ㅇㅇ
#### - Vector DB 개념
#### - 간단한 Q&A 만들기
#### - RAG 고도화
#### - Chunking 전략
#### - 출처 표시
#### - 검색 품질 평가
#### - ./embedding
  - 특징 추출 자동화 (Feature Engineering)
    - 딥러닝 모델(Word2Vec)을 사용하여 컴퓨터가 스스로 데이터 사이의 관계를 찾아 특징을 숫자로 추출.
  - 고차원 공간으로의 투영(Vectorization)
    - 텍스트(문자를) 딥러닝 연산이 가능한 '수치(Vector)'로 변환.
    - vector_size =50 설정을 통해, "짜장면"이라는 단어를 50개의 서로 다른 특징을 가진 좌표로 변환하여 50차원의 가상 공간에 배치.
    - 이 과정에서 비슷한 의미를 가진 단어들이 공간상에서 가깝게 모이도록 신경망의 가중치(weight)를 업데이트.
  - 비지도 학습 기반의 사전 훈련(Pre-training)
    - Label 따로 없는 상태에서 문장 데이터만으로 학습. (복잡한 딥러닝 모델을 만들기 위한 전처리 과정)
  - 해당모델을 통해 elasticsearch의 검색필터를 활용해보기 ..
  -   
  - 지금은 학습된 단어 이외의 단어는 에러가남..(에러처리 혹은 단어를 학습시키면 됨.) 여기서.. 한단계 더 올리기 위해서는..?

## 3. How to make AI take actions (Agents)
#### - Tool Calling

## 4. How to fine-tune LLMs

## 5. How to build LLMs from scratch


