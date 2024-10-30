import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 포트폴리오 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 무엇인가요?",
    "모델은 어떤걸 썼나요?",
    "프로젝트 인원은 어떻게 되나요?",
    "프로젝트 기간은 어떻게 되나요?",
    "조장이 누구인가요?",
    "데이터는 뭘 이용했나요?",
    "프로젝트 하는데 어려움은 없었나요?"
]

answers = [
    "이미지를 활용해 피부 상태 측정을 통한 화장품 추천입니다.",
    "cnn 모델을 사용하였습니다.",
    "김준희,문혜선 총 2명으로 진행하였습니다.",
    "총 3주입니다. 주제 선정, 모델 구현, ppt 및 발표 준비 등으로 구성했습니다.",
    "조장은 김준희가 맡아서 하였습니다.",
    "AI HUB 에 있는 피부 상태 데이터를 사용하였습니다.",
    "아무래도 2명이서 진행하다보니 맡아서 할 부분이 많아서 조금 힘들었고 데이터를 이해하는데 어려움이 조금 있었습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("포트폴리오 소개 챗봇")
# 이미지 표시
st.image("skin.jpg", caption="Welcome to the Restaurant Chatbot", use_column_width=True)

st.write("저희가 만든 포트폴리오에 대해서 질문해주세요. 예: 주제가 무엇인가요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
