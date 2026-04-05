import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline # 허깅페이스 로컬 모델을 Langchain이 알아들을 수 있게 포장해 주는 도구
from typing import TypedDict # 파이썬의 타입 힌트 도구로, 에이전트의 상태 구조를 명확히 정의하기 위해 사용
from langgraph.graph import StateGraph, START, END # LangGraph의 핵심인 그래프, 시작점, 종료점을 불어옴

# 1. 허깅페이스 로컬 모델 로딩(month5와 동일)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

quantization_config = BitsAndBytesConfig(
    # 모델의 파라미터 한개의 크기를 16비트에서 4비트로 쪼개서 가져옴
    # 이 옵션을 켜는 순간 모델의 덩치가 4GB가 되기 때문에 1/4로 확 줄어듬
    load_in_4bit = True,
    
    #저장할 땐 4비트로 작게 저장하되, 연산(Compute)를 할때는
    #16비트로 넓게 펴서 계산하라
    #4비트인 채로 수학 계산을 하면 너무 멍청해지기 때문에, 계산하는 그 찰나의 순간에만
    #잠시 16비트로 복원해서 모델의 똑똑한 지능을 유지하는 실무 최적화 기법

    bnb_4bit_compute_dtype = torch.float16,

    #"nf4"는 NormalFloat4의 약자
    # 일반적인 4비트가 아니라, 파라미터들이 보통 '정규분포(종모양)'을 띈다는 수학적 사실을 이용해,
    # 모델의 파라미터 손상을 최소화하면서 정규분포를 따르는 4비트로 압축하는 최신 기술
    
    bnb_4bit_quant_type = "nf4",


    #이중 양자화(Double Quantization)
    #모델을 1차로 압축하고 나면 압축 비율(Scale factor)라는 찌꺼기가 생김
    #이 찌꺼기 조차도 메모리를 차지하므로, 이것까지 한번 더 (Double) 압축해서
    #극한의 메모리 절약을 달성하는 옵션

    bnb_4bit_use_double_quant = True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = quantization_config, # 위에서 만든 4-bit 압축기를 모델에 장착
 
    device_map = "auto" # 수동으로 .to(device) 할 필요 없이, 자동으로 GPU 사용 명시 
)

print("Mistral 7B가 4090 GPU에 안전하게 안착했습니다.")

allocated_memory = torch.cuda.memory_allocated() / (1024**3)
print(f"현재 GPU VRAM 사용량: {allocated_memory:.2f} GB / 24.00 GB")

# 2. LangChain 파이프라인으로 포장 (Wrapping)
# 모델을 직접 쓰기 쉽게 텍스트 생성 전용 파이프라인으로 만듦

text_generation_pipeline = pipeline(
    "text-generation", # 텍스트 생성 작업임을 명시
    model = model, # 방금 로딩한 4-bit 양자화된 모델을 파이프라인에 같이 올림
    tokenizer = tokenizer, # 짝꿍인 토크나이저도 같이 올림
    max_new_tokens = 100, # 모델이 생성할 최대 단어 수를 100개로 제한 (쓸데 없이 길게 말하는 것 방지)
    do_sample = False, # 창의성(랜덤성)을 끄고 가장 확률이 높은 모범 답안만 뱉게 만듦
    return_full_text = False # 내 질문을 앵무새처럼 따라하는 것을 막고, 새로 만든 답변만 딱 잘라서 반환
)

# 위에서 만든 텍스트 생성 전용 파이프라인을 Langchain 파이프라인에 딱 맞게 포장(wrapping)
llm = HuggingFacePipeline(pipeline = text_generation_pipeline)

# 3. LangGraph 세팅 : 상태(State)와 노드(Node) 정의
# 파이썬 3.5부터 변수명: 데이터타입 이라는 문법을 새로 만들었고, 변수의 타입을 정적으로 결정할 수 있게 됨
# 에이전트들이 대화 기록을 공유할 '공용 메모장' 형태를 정의
# 앞으로 우리 에이전트들이 공용으로 쓸 메모장(AgentState) 규칙을 정할게! 
# 이 메모장은 딕셔너리(TypedDict) 형태인데, 
# 안에는 반드시 messages라는 이름의 칸이 있어야 하고, 
# 그 칸 안에는 무조건 '텍스트(str)'만 적을 수 있어. 딴 거 적으면 에러 낼 거야!
class AgentState(TypedDict):
    messages: str

# 에이전트(Node) 정의: 각자의 역할(에이전트가 자기턴이 왔을때 수행할 행동, 함수) 부여
# state라는 매개변수의 타입을 AgentState을 정적으로 정의
def simple_agent(state: AgentState):
    print("에이전트가 답변을 생성 중입니다.")
    
    # Mistral Instruct 모델일 말을 잘 듣도록 [INST] [/INST] 형식으로 프롬프트를 조립
    prompt = f"[INST] {state['messages']} [/INST]"
    # 조립된 프롬프트를 LLM에 던져서 답변을 뽑아냄
    response = llm.invoke(prompt)
    # 뽑아낸 답변의 앞뒤 지저분한 공백을 자르고, 메모장에 담아서 다음 턴으로 넘김
    return {"messages": response.strip()}

# 4. 그래프(Graph) 조립 및 컴파일

workflow = StateGraph(AgentState) # 위에서 만든 메모장(AgentState) 규칙을 따르는 새로운 작업 흐름(그래프)를 염

workflow.add_node("agent", simple_agent) # "agent"란 이름표를 달고, simple_agent 함수를 수행하는 노드를 그래프에 배치

workflow.add_edge(START, "agent") # 시스템이 시작(START)되면 무조건 "agent" 노드부터 실행되도록 선을 그음
workflow.add_edge("agent", END) # "agent" 노드의 작업이 끝나면 전체 프로세스가 종료(END) 되도록 선을 그음

app = workflow.compile() # 우리가 그린 이 그래프를 실제 실행 가능한 애플리케이션으로 꽉 뭉침(빌드)

# 테스트 코드
if __name__ == "__main__":
    test_question = "What is the capital of France? Only answer with one word." # 테스트용 질문을 준비합니다
    print(f"\n🗣️ 사용자 질문: {test_question}") 
    
    # ★매우 중요★
    # 컴파일된 앱에 초기 상태({'messages': 질문})를 넣고 실행버튼(invoke)을 누릅니다!
    # 이 초기 상태가 START->"agent"->END 거쳐서 빠져나옴
    result = app.invoke({"messages": test_question})
    
    # 그래프를 싹 다 돌고 최종적으로 튀어나온 결과물의 'messages' 칸을 확인해서 출력합니다
    print(f"🤖 에이전트 답변: {result["messages"]}")