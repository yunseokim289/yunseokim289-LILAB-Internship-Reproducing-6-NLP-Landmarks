import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. 사용할 모델 id 작성
# 형식: "만든사람/모델이름"
# 7B: B는 Billion의 약자고 10억을 의미, 총 70억개의 파라미터(weight) 보유중인 모델
# 파라미터 개수 x 2를 한게 이 모델을 돌리기 위해 필요한 최소 GPU 메모리(GB = bit)
# 이 모델은 7(B) x 2 = 14 GB의 최소 GPU 메모리을 가짐
# 고로 파라미터 1개의 크기 : 14-bit (14 GB에서 14는 동일)
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# 2. 4-bit 압축기 설정
# 거대한 16GB짜리 모델을 4090(24GB) GPU 메모리에 널널하게 올리기 위한 핵심 세팅

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

# 3. 토크나이저 로드
# 모델이 글자를 읽을 수 있도록 영어/한국어 문장을 숫자로 쪼개주는 번역기

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4. 4-bit 양자화 적용해서 모델 로드

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = quantization_config, # 위에서 만든 4-bit 압축기를 모델에 장착
 
    device_map = "auto" # 수동으로 .to(device) 할 필요 없이, 자동으로 GPU 사용 명시 
)

print("Mistral 7B가 4090 GPU에 안전하게 안착했습니다.")

# 5. 메모리 다이어트 결과 확인
# -------------------------------------------------------------------
# torch.cuda.memory_allocated()는 현재 GPU가 사용 중인 메모리를 바이트(Byte) 단위로 알려줍니다.
# 이걸 1024의 3제곱(1024 * 1024 * 1024)으로 나누면 우리가 보기 편한 기가바이트(GB) 단위가 됩니다.
allocated_memory = torch.cuda.memory_allocated() / (1024**3)
print(f"📊 현재 GPU VRAM 사용량: {allocated_memory:.2f} GB / 24.00 GB")


# 6. 0-shot (zere-shot) prompting

# 테스트할 영화 리뷰 문장(뉘앙스가 헷갈리는 까다로운 부정 리뷰)
review_text = "The acting was okay, but the plot made absolutely no sense and I fell asleep."

# 0-shot prompting 
# Mistral 모델은 [INST] 와 [/INST] 사이에 지시사항을 적어주면 찰떡같이 알아들음
prompt = f"""[INST] Classify the sentiment of the following movie review as either 'Positive' or 'Negative'. Only answer with one word.

Review: {review_text}
Sentiment : [/INST]
"""
print("0-shot prompt로 모델이 답변을 고민 중입니다.")

# 6-1. 텍스트 프롬프트를 숫자로 쪼개서 (Tokenize) GPU로 보냄
# return_tensors : 결과물을 파이썬 리스트가 아닌 파이토치의 텐서 형태로 변환해서 리턴
inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")

# 6-2. 모델 추론
with torch.no_grad(): # 파인튜닝(학습)이 아니니까 기울기 계산 끄기(메모리 절약)
    outputs = model.generate(
        **inputs,
        max_new_tokens = 10, # 10글자 까지만 허용
        do_sample = False, # 창의성 끔 (진지하고 일관된 답변을 냄)
        pad_token_id = tokenizer.eos_token_id # 경고 메시지 방지용
    )

# 6-3. 모델이 뱉어낸 숫자들을 다시 사람이 읽을 수 있는 영어로 번역 (Decode)
# skip_special_tokens = True : 쓸데없는 기호는 빼고 주라고 지시
response = tokenizer.decode(outputs[0], skip_special_tokens = True)

# 6-4. 프롬프트 부분을 잘라내고 모델이 순수하게 대답한 부분만 추출
# 우리가 쓰는 Mistral, Llama, GPT 같은 모델들은 기본적으로 "이전 단어들을 보고 다음 단어를 이어붙이는(Causal LM)" 원리로 작동합니다.
# 그래서 모델에게 결과를 달라고 하면, 모델은 자기가 새로 만들어낸 정답만 딱! 주는 게 아니라, 
# 우리가 처음에 던져줬던 길고 긴 프롬프트 질문을 그대로 앵무새처럼 한 번 다 읊은 다음에 
# 자기 대답을 덧붙여서 줌
answer = response.split('[/INST]')[-1].strip()

print(f"0-shot 리뷰 : {review_text}")
print(f"0-shot 모델의 정답 예측 : {answer}")

# 7. 1-shot (one-shot) prompting
 
review_text = "The acting was okay, but the plot made absolutely no sense and I fell asleep."

# 1-shot prompting
# 지시사항 바로 아래에 [정답이 포함된 완벽한 예시] 하나를 슬쩍 끼워 넣음

prompt_1_shot = f"""[INST] Classify the sentiment of the following movie review as either 'Positive' or 'Negative'. Only answer with one word.

Review: This movie was absolutely fantastic! I loved every minute of it.
Sentiment : Positive

Review: {review_text}
Sentiment : [/INST]
"""
print("1-shot prompt로 모델이 답변을 고민 중입니다.")

# 7-1. 텍스트 프롬프트를 숫자로 쪼개서 (Tokenize) GPU로 보냄
# return_tensors : 결과물을 파이썬 리스트가 아닌 파이토치의 텐서 형태로 변환해서 리턴
inputs = tokenizer(prompt_1_shot, return_tensors = "pt").to("cuda")

# 7-2. 모델 추론
with torch.no_grad(): # 파인튜닝(학습)이 아니니까 기울기 계산 끄기(메모리 절약)
    outputs = model.generate(
        **inputs,
        max_new_tokens = 10, # 10글자 까지만 허용
        do_sample = False, # 창의성 끔 (진지하고 일관된 답변을 냄)
        pad_token_id = tokenizer.eos_token_id # 경고 메시지 방지용
    )

# 7-3. 모델이 뱉어낸 숫자들을 다시 사람이 읽을 수 있는 영어로 번역 (Decode)
# skip_special_tokens = True : 쓸데없는 기호는 빼고 주라고 지시
response = tokenizer.decode(outputs[0], skip_special_tokens = True)

# 7-4. 프롬프트 부분을 잘라내고 모델이 순수하게 대답한 부분만 추출
# 우리가 쓰는 Mistral, Llama, GPT 같은 모델들은 기본적으로 "이전 단어들을 보고 다음 단어를 이어붙이는(Causal LM)" 원리로 작동합니다.
# 그래서 모델에게 결과를 달라고 하면, 모델은 자기가 새로 만들어낸 정답만 딱! 주는 게 아니라, 
# 우리가 처음에 던져줬던 길고 긴 프롬프트 질문을 그대로 앵무새처럼 한 번 다 읊은 다음에 
# 자기 대답을 덧붙여서 줌
answer = response.split('[/INST]')[-1].strip()

print(f"1-shot 리뷰 : {review_text}")
print(f"1-shot 모델의 정답 예측 : {answer}")

# 8. 5-shot (five-shot) prompting 

review_text = "The acting was okay, but the plot made absolutely no sense and I fell asleep."

# 5-shot prompting
# 지시사항 바로 아래에 [정답이 포함된 완벽한 예시] 5개를 끼워 넣음

prompt_5_shot = f"""[INST] Classify the sentiment of the following movie review as either 'Positive' or 'Negative'. Only answer with one word.

Review: This movie was absolutely fantastic! I loved every minute of it.
Sentiment: Positive

Review: A complete waste of time. The worst movie I've seen this year.
Sentiment: Negative

Review: Brilliant acting and a captivating storyline.
Sentiment: Positive

Review: The dialogue was incredibly cringey and unnatural.
Sentiment: Negative

Review: A masterpiece of modern cinema. Highly recommended!
Sentiment: Positive

Review: {review_text}
Sentiment : [/INST]"""

print("5-shot 프롬프트로 모델이 답변을 고민 중입니다.")

# 8-1. 텍스트 프롬프트를 숫자로 쪼개서 (Tokenize) GPU로 보냄
# return_tensors : 결과물을 파이썬 리스트가 아닌 파이토치의 텐서 형태로 변환해서 리턴
inputs = tokenizer(prompt_5_shot, return_tensors = "pt").to("cuda")

# 8-2. 모델 추론
with torch.no_grad(): # 파인튜닝(학습)이 아니니까 기울기 계산 끄기(메모리 절약)
    outputs = model.generate(
        **inputs,
        max_new_tokens = 10, # 10글자 까지만 허용
        do_sample = False, # 창의성 끔 (진지하고 일관된 답변을 냄)
        pad_token_id = tokenizer.eos_token_id # 경고 메시지 방지용
    )

# 8-3. 모델이 뱉어낸 숫자들을 다시 사람이 읽을 수 있는 영어로 번역 (Decode)
# skip_special_tokens = True : 쓸데없는 기호는 빼고 주라고 지시
response = tokenizer.decode(outputs[0], skip_special_tokens = True)

# 8-4. 프롬프트 부분을 잘라내고 모델이 순수하게 대답한 부분만 추출
# 우리가 쓰는 Mistral, Llama, GPT 같은 모델들은 기본적으로 "이전 단어들을 보고 다음 단어를 이어붙이는(Causal LM)" 원리로 작동합니다.
# 그래서 모델에게 결과를 달라고 하면, 모델은 자기가 새로 만들어낸 정답만 딱! 주는 게 아니라, 
# 우리가 처음에 던져줬던 길고 긴 프롬프트 질문을 그대로 앵무새처럼 한 번 다 읊은 다음에 
# 자기 대답을 덧붙여서 줌
answer = response.split('[/INST]')[-1].strip()

print(f"5-shot 리뷰 : {review_text}")
print(f"5-shot 모델의 정답 예측 : {answer}")