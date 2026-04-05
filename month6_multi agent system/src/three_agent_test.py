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
    max_new_tokens = 150, # 모델이 생성할 최대 단어 수를 150개로 제한 (쓸데 없이 길게 말하는 것 방지)
    do_sample = False, # 창의성(랜덤성)을 끄고 가장 확률이 높은 모범 답안만 뱉게 만듦
    return_full_text = False # 내 질문을 앵무새처럼 따라하는 것을 막고, 새로 만든 답변만 딱 잘라서 반환
)

# 위에서 만든 텍스트 생성 전용 파이프라인을 Langchain 파이프라인에 딱 맞게 포장(wrapping)
llm = HuggingFacePipeline(pipeline = text_generation_pipeline)

# 3. 상태(State) 정의: 두 에이전트(Node)가 공유할 '공용 메모장' 정의
# single_agent_test.py에서는 messages 칸만 있었지만, 이제는 주제, 초안, 최종 리뷰 칸이 나뉘어 있음
class AgentState(TypedDict):
    topic: str # 사용자가 던져준 글쓰기 주제
    research_data: str # Researcher가 조사해 온 자료를 담을 새로운 칸
    draft: str # Writer 에이전트가 작성한 초안
    review: str # Reviewer 에이전트가 작성한 평가 및 피드백
    iteration: str # 이번이 몇 번째 수정인지 기록하는 카운터

# 4. 에이전트(Node) 정의: 각자의 역할(에이전트가 자기턴이 왔을때 수행할 행동, 함수) 부여

# 4-1. 자료 조사원(Researcher) 에이전트
def researcher_agent(state: AgentState):
    print("Researcher 에이전트가 주제에 대한 핵심 자료를 조사하고 있습니다.")
    
    # 모델에게 주제에 대한 2가지 핵심 사실을 짧게 뽑아달라고 지시
    prompt = f"[INST] Provide 2 key facts about {state['topic']}. Keep it brief and objective. [/INST]"
    response = llm.invoke(prompt)

    # 조사된 자료를 공용 메모장의 'research_data' 칸에 적어서, 반복 횟수를 0으로 초기화화여 다음 에이전트에게 넘김
    return {"research_data": response.strip(), "iteration": 0}


# 4-2. 작성자(Writer) 에이전트
def writer_agent(state: AgentState):
    iteration = state.get("iteration", 0) # state의 "iteration" 키가 있으면 그 값 반환, 없으면 0을 반환

    if iteration == 0:
        print("Writer 에이전트가 Researcher의 조사 자료를 바탕으로 첫번째 초안을 작성하고 있습니다.")
        # 주어진 주제에 대해 2문장짜리 짧은 소개글을 Researcher가 모아온 팩트를 기반으로 작성하라고 지시
        prompt = f"[INST] Write a short, 2-sentence introduction about {state['topic']} using ONLY these facts: {state['research_data']} [/INST]"
    else:
        print(f"Writer 에이전트가 Reviewer의 피드백을 반영하여 {iteration + 1} 번째 수정을 진행합니다.")
        # 피드백을 반영해서 다시 쓰라고 지시
        prompt = f"""[INST] Revise the following draft about {state['topic']} 
        based on this feedback: {state['draft']}, Feedback: {state['review']}
        Provide the revised 2-sentense draft only. /INST]
        """

    response = llm.invoke(prompt)
    # 자신이 쓴 글을 공용 메모장의 'draft' 칸에 적어서 Reviewer 에이전트에게 넘김
    return {"draft": response.strip()}

# 4-3. 검토자(Reviewr) 에이전트
def reviewer_agent(state: AgentState):
    print("Reviewer 에이전트가 초안을 검토하고 피드백을 남깁니다.")
    # Writer가 넘겨준 초안(state['draft'])을 읽고, 1문장짜리 냉철한 평가를 하라고 지시
    prompt = f"""[INST] Review the following draft about {state['topic']}: {state['draft']}.
    Provide a 1-sentence critique or feedback on this draft. [/INST]
    """
    response = llm.invoke(prompt)

    current_iteration = state.get("iteration", 0) # state의 "iteration" 키가 있으면 그 값 반환, 없으면 0을 반환
    # 평과 결과를 공용 메모장의 'review' 칸에 적어서 최종 결과물 리턴
    return {"review": response.strip(), "iteration": current_iteration + 1}

# 5. 조건부 라우터(Router) 함수
# Reviewer 검토를 마친 후, 공용 메모장을 어디로 보낼지 결정하는 라우터
def conditional_router(state: AgentState):
    # 반복 횟수가 2번 미만이면 다시 Writer에게 돌려보냄(빠꾸)
    if state.get("iteration", 0) < 2:
        print("아직 부족합니다! Writer에게 다시 돌려보냅니다.")
        return "Writer"

    # 2번을 꽉 채웠으면 최종 승인하고 끝냄
    else:
        print("훌륭합니다! 최종 승인하여 프로세스를 종료합니다.")
        return END



# 6. 그래프(Graph) 조립 및 컴파일

workflow = StateGraph(AgentState) # 위에서 만든 공용 메모장(AgentState) 규칙을 따르는 새로운 작업 흐름(그래프)를 생성

# 3명의 에이전트를 그래프에 배치
workflow.add_node("Researcher", researcher_agent) # "Researcher"란 이름표를 달고, researcher_agent 함수를 수행하는 에이전트(노드)를 그래프에 배치
workflow.add_node("Writer", writer_agent) # "Writer"란 이름표를 달고, writer_agent 함수를 수행하는 에이전트(노드)를 그래프에 배치
workflow.add_node("Reviewer", reviewer_agent) # "Reviewer"란 이름표를 달고, reviewr_agent 함수를 수행하는 에이전트(노드)를 그래프에 배치

# 작업 순서대로 화살표(Edge)를 그음
workflow.add_edge(START, "Researcher") # 시스템이 시작(START)되면 무조건 "Researcher" 노드부터 실행되도록 선을 그음
workflow.add_edge("Researcher", "Writer") # Researcher가 자기의 역할을 다 하면 Writer에게 리턴값을 전달 (★멀티 에이전트의 핵심)
workflow.add_edge("Writer", "Reviewer") # Writer가 자기의 역할을 다 하면 Research에게 리턴값을 전달 (★멀티 에이전트의 핵심)
workflow.add_conditional_edges("Reviewer", conditional_router) # "Reviewer" 노드의 작업이 끝나면 무조건 END로 가는게 아니라, Router함수의 결정에 따라 갈 길을 정함

app = workflow.compile() # 우리가 그린 이 그래프를 실제 실행 가능한 애플리케이션으로 꽉 뭉침(빌드)

# 5. 테스트 코드
# ---------------------------------------------------------
if __name__ == "__main__":
    test_topic = "Artificial Intelligence in the future"
    print(f"\n🗣️ 사용자 지시: '{test_topic}'에 대해 완벽해질 때까지 수정해라!")
    
    # ★매우 중요★
    # 컴파일된 앱에 'topic' 칸만 채운 상태({"topic": test_topic})를 넣고 실행버튼(invoke)을 누릅니다!
    # 'draft'와 'review'칸은 멀티에이전트가 채울 예정
    # 이 상태가 START->"Research"-> "Writer"->"Reviewer"->conditional_router 거쳐서 빠져나옴
    result = app.invoke({"topic": test_topic})

    # 그래프를 싹 다 돌고 최종적으로 튀어나온 결과물의 'topic', 'research_data, 'draft', 'iteraton', 'review' 칸을 확인해서 출력합니다
    
    print(f"주제: {result['topic']}")
    print(f"참고한 기초 팩트: {result['research_data']}")
    print(f"Writer의 초안:{result['draft']}")
    print(f"총 반복 횟수: {result['iteration']}회")
    print(f"Reviewer의 피드백:{result['review']}")