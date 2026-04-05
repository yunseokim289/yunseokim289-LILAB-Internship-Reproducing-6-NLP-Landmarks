# LILAB Intern Project
[MONTH 1]
## Week 1: Environment Setup & Sanity Check
- Environment: Python 3.10, PyTorch 2.5.1 installed.
- Repository Structure: Created `src`, `configs`, `scripts`, `data`, `runs` folders.
- Sanity Check: Successfully ran `train.py` with Mixed Precision (AMP) enabled.
- Result: `Sanity Check 완료! Loss: -456.4113` (Run on CPU).

## 2주 차 진행 상황
- Step 1: 데이터 전처리 및 단어장 구축 완료
- 원문 데이터(`toy_data.txt`) 로드 및 소문자 정규화 수행 
- Hyperparameter: `min_freq=5` 적용하여 희귀 단어 제거
- Result: 단어장 크기(V) 3324개, 전체 학습 데이터(T) 약 2243862개 확보

## 3주 차 진행 상황 : 데이터셋 구축 및 서브샘플링 (Subsampling)

### 1. 고빈도 단어 서브샘플링 (Subsampling)
- 구현 내용: 
  - 학습 효율을 저해하는 고빈도 단어(예: 'the', 'a' 등)를 확률적으로 제거하기 위해 삭제 확률 
    P(w_i) = 1 - \sqrt{t/f(w_i)} 수식을 적용 
  - 논문에서 권장하는 임계값(Threshold)인 t = 1e-5를 하이퍼파라미터로 설정하였습니다. 
- 기대 효과: 전체 토큰 수를 최적화하여 학습 속도를 높이고, 희소 단어(Rare words)에 대한 벡터 표현 품질을 개선

### 2. 학습용 데이터 쌍 생성 (Skip-gram)
- 기법: 슬라이딩 윈도우 (Sliding Window) 적용 
- 구현 상세: 
  - 윈도우 크기를 5로 설정하여 중심 단어와 주변 단어의 관계를 (Input, Output) 형태의 인덱스 쌍으로 변환
  - 재현성: 실험 결과의 일관성을 위해 `random.seed(42)`를 사용하여 샘플링 결과를 고정하였습니다. 
- 최종 검증(Sanity Check) 결과:
  - Original Length: 2,243,862개
  - Subsampled Length:  304281개
  - Total Training Pairs: 1219070개
  - Sample (Input, Output): [(5, 6), (5, 13), (6, 5)]

## 4주 차 진행 상황 : Skip-gram 모델 구현 및 Negative Sampling 이식

### 1. Skip-gram 아키텍처 설계
- 구현 내용: 
  - 논문의 skip-gram 구조를 정밀하게 이식하여 중심 단어(`in_embed`)와 주변 단어(`out_embed`) 테이블 분리 구현
  - 가중치 초기화: `in_embed`는 Uniform Distribution, `out_embed`는 Zero Initialization을 적용하여 논문 표준 준수
- 기대 효과: 단어 간의 유사도를 벡터 공간상의 거리로 학습할 수 있는 신경망 기반을 구축하고 초기 학습 안정성 확보

### 2. Negative Sampling 및 커스텀 손실 함수
- 기법: Negative Sampling (NEG) 직접 구현
- 구현 상세: 
  - `torch.bmm` 및 `unsqueeze` 연산을 활용하여 배치(Batch) 단위의 효율적인 병렬 내적 연산 로직 구현
  - `F.logsigmoid`를 적용하여 수식의 로그-시그모이드 연산에 대한 수치적 안정성 확보
- 최종 검증(Sanity Check) 결과:
  - 실험 환경: Batch Size 4, Embedding Dim 10, Negative Samples 5
  - Shape Check: Positive Score `torch.Size([4])`, Negative Score `torch.Size([4, 5])` 일치 확인
  - Result: `✅ Step 3: 모델 및 손실 함수 연산 테스트 통과!` (Final Loss 약 4.1589 산출)

## 5주 차 진행 상황 : 대규모 모델 학습 및 최적화 

### 1. GPU 가속 및 연산 최적화
- 실험 환경: NVIDIA GeForce RTX 4090 GPU 서버 활용
- 구현 상세:
  - AMP(Automatic Mixed Precision) 적용: FP16 연산을 수행하여 학습 속도 및 메모리 효율 최적화
  - 데이터 파이프라인: (Center, Pos, Neg) 3종 세트 구성

### 2. 학습 수행 및 결과
- Hyperparameter: `Batch Size=4096`, `Learning Rate=0.001`, `Epochs=5`, `Embedding Dim=100`
- Result:
  - Final Loss: 0.3194 (최종 에폭 완료 및 `word2vec_model.pth` 저장 완료)

## 6주 차 진행 상황 : Harry Potter Word2Vec Project

본 프로젝트는 해리포터 영화 시리즈 텍스트 데이터를 바탕으로 단어 임베딩 모델(Word2Vec)을 구축하고, 모델이 단어 사이의 논리적/문맥적 관계를 얼마나 잘 이해하는지 평가합니다.

## 평가 결과 (Evaluation)
### 1. 유사도 테스트 (Similarity)
- `emma` 입력 시 출력 시 emma와 가장 유사한 단어(top-5) 출력

### 2. 유추 테스트 (Analogy)
- 질문: harry : potter = daniel : ?
- 결과: radcliffe
- 분석: 이름-성 관계 도출

## 7주차 진행 상황 : Word2Vec Embedding Visualization

본 문서는 학습된 Word2Vec 모델의 임베딩 벡터를 시각적으로 분석한 결과를 기록합니다.

## 1. 시각화 개요
- 목적: 100차원의 단어 벡터를 2차원 평면으로 투영하여, 단어 간의 의미적 거리와 군집(Cluster) 형성 여부를 육안으로 검증함.
- 방법: PCA (Principal Component Analysis) 기법을 사용하여 차원 축소.
- 대상 단어: `harry`, `ron`, `hermione` 등 주요 등장인물 및 `movie`, `director` 등 일반 명사.

## 2. 분석 결과 (Key Findings)
- 군집화(Clustering):
  -"harry","potter" and "emma", "watson"처럼 비슷한 단어들끼리 군집화를 이룸
  -사람 이름들(harry, daniel, emma 등)은 한쪽에 뭉쳐 있고, movie, director, magic,  --hogwarts 같은 일반 명사들은 그들과 조금 떨어진 다른 쪽에 뭉쳐 있음.
  이는 모델이 사람과 사물/개념의 의미적 차이를 구분하고 있다는 증거

결과 : src/word_embedding_plot.png

[MONTH 2]

# 1주차 : Harry Potter Language Modeling with LSTM

본 프로젝트는 ELMo 논문의 전방향 언어 모델 수식을 직접 구현하고 학습시킨 결과물입니다.

## Final Training Progress
| Epoch | Training Loss 
| 1 | 7.0521 | Initial State 
| 20 | 0.4748| Fully Converged 


## Artifacts
- runs/exp_0124_1343/log.txt 에 모든 학습 지표 기록 완료.
- runs/exp_0124_1343/model.pth` 에 학습된 파라미터 저장 완료.

## Conclusion
모델이 20 에폭 동안 손실을 개선하며 성공적으로 학습되었습니다.

# 2주차: 추론 

2주차에는 학습된 가중치를 로드하여 다음 단어를 예측하는 추론 시스템을 구축했습니다.

## 최근 업데이트
- 추론 파이프라인 완성: 저장된 가중치(`model.pth`)를 불러와 새로운 입력을 처리하는 로직 구현.
- 결과 관리 자동화: 추론 시마다 `runs/` 디렉토리에 실행 시간별 로그 및 결과 자동 저장.
- 실행 스크립트 최적*: `./scripts/gen_rnn.sh`를 통한 간편한 실행 체계 마련.

## 추론 테스트 결과 (검증 단계)
- 입력 시드(Seed): 토큰 ID `2`
- 모델 예측: 토큰 ID `808`
- 설정: 탐욕적 탐색 (Greedy Search / argmax 방식)

## 3주차 : 실제 데이터를 전처리 후 모델 학습 및 추론

### 1. Data Preprocessing
- Tokenization: 영문 소문자 변환 및 특수문자 제거
- Vocabulary: `Vocab` 클래스를 통해 단어-인덱스 매핑 (`stoi`, `itos`)
- Unknown Token: 희귀 단어 처리를 위한 `<unk>` 토큰 적용

### 2. Model Architecture (LSTM)
- Embedding Layer: 단어를 고차원 벡터로 변환
- LSTM Layer: 문맥(Context) 학습 및 장기 의존성(Long-term Dependency) 문제 해결

### 3. Training & Inference
- Loss Function: CrossEntropyLoss
- Optimization: Adam Optimizer
- Inference: Argmax 기반의 Next Token Prediction (Recursive Generation)

## result
Final Loss: 2.1581 (Epoch 20)
Generated Sample: "harry s wand had done so far it was a <unk> and the one he said stupidly there are more people like that said ron sharply...

## 4주차 : 다양한 생성 전략(Decoding) 적용 및 텍스트 품질 개선
1. Advanced Decoding Strategies
-Temperature Scaling: 온도를 적용하여 점수 조절
-Top-k Sampling: 누적 확률과 상관없이 상위 k개의 후보 단어만 추출하여 필터링
-Top-p Sampling: 누적 확률 분포가 p에 도달하는 최소한의 단어 집합만 후보로 선택하여 동적 필터링 적용
2. Sampling & Generation
-Multinomial Sampling: Argmax의 단조로움을 극복하기 위해 확률 분포에 기반한 랜덤 샘플링 적용
-Recursive Generation: 생성된 단어를 다시 입력으로 사용하는 재귀적 방식을 유지하며 최적의 서사 구조 생성
3. result
-Hyperparameters: Temperature=0.8, Top-k=50, Top-p=0.9
-Generated Sample: "the door opened harry and hermione entered the great hall for breakfast at the end of the corridor when the archway


[MONTH3]
# [Month 3] Week 1: Multi-Head Attention & Causal Masking

## Project Overview
Attention Is All You Need(Vaswani et al., 2017) 논문의 핵심인 Multi-Head Attention을 구현하고, GPT(Decoder-only) 구조에 필수적인 Causal Masking 기능을 검증했습니다.

### Key Features
1.  Multi-Head Attention (Section 3.2.2)
    - Multi-Head Attention : MultiHead(Q, K, V) == Concat(head1, ...) W^O 구현 후 리턴
2.  Causal Masking (Section 3.2.3)
    - 미래 시점의 토큰을 `-1e9`로 마스킹하여 참조를 원천 차단.
    - `Softmax` 적용 시 미래 토큰의 확률이 정확히 0이 됨을 보장.

#[Month 3] Week 2: Positional Encoding & Transformer Block 
## Project Overview
Transformer 모델이 단어의 순서를 인식할 수 있도록 Positional Encoding을 구현하고, Attention과 Feed-Forward Network를 결합하여 모델의 기본 단위인 Transformer Block을 완성했습니다.
### Key Features
1. Positional Encoding (Section 3.5)
  -Sinusoidal Encoding: 사인(sin)과 코사인(cos) 함수를 이용해 주기적인 위치 정보를 생성 
2. Position-wise Feed-Forward Networks (Section 3.3)
  -Structure: 두 개의 선형 변환(Linear) 사이에 ReLU 활성화 함수를 포함한 구조 
3. Residual Connection & Layer Normalization (Section 3.1)
  -Add & Norm: LayerNorm(x + Sublayer(x)) 구조를 적용하여 정보 손실 방지(Residual) 및 학습    안정화(Norm).
  -Dual Application: Self-Attention 후와 FFN 후, 총 두 번에 걸쳐 "Add & Norm"을 적용하여 블록의 완성도 확보.

# [Month 3] Week 3: GPT Model Assembly (The Architecture)

## Project Overview
단일 Transformer Block을 N번 적층(Stacking)하고, Embedding Layer(입력)와 Linear Head(출력)를 결합하여 GPT 모델의 전체 아키텍처(Full Architecture)를 완성했습니다.

## Key Features

1. Full Transformer Architecture Implementation
   - Embedding Layer: 입력 토큰을 벡터로 변환하고 위치 정보를 결합.
   - Stacking encoder/decoder layers: `nn.ModuleList`를 사용하여 'EncoderLayer', 'DecoderLayer'을 N층으로 깊게 쌓아올린 구조 구현.
   - Output Head: 다음단어의 확률(Logits)을 예측.

2. Integration Test
   - 입력 텐서 `(Batch, Seq_Len)`가 모델을 통과한 후 `(Batch, Seq_Len, Vocab_Size)` 형태로 정확하게 출력됨을 검증 완료.

3. Encoder Implementation
   - Bidirectional Attention: 미래 정보를 볼 수 없는 Decoder와 달리, 문장 전체의 문맥을 동시에 참조할 수 있는 마스크 없는 Self-Attention 구조 구현.
   - Encoder Stack: 입력 문장(Source Sentence)의 의미적 특징(Context Vector)을 추출하여 Decoder에 전달하는 역할 수행.

4. Decoder Implementation
   - Masked Self-Attention: 타겟 문장(Target Sentence) 학습 시, 미래의 단어를 미리 보고 컨닝하지 못하도록 상삼각행렬 마스크를 적용
   - Dual Attention Structure: 자신의 문맥을 보는 Self-Attention과 원문을 참조하는 Cross-Attention을 순차적으로 수행하는 이중 어텐션 구조 구현.

5. Cross-Attention Mechanism
   - Interaction: Decoder가 Encoder의 출력(Key, Value)을 참조하는 Cross-Attention Layer 추가.
  


# [Month 4]: BERT Pretraining & Fine-Tuning Pipeline

## Project Overview
본 프로젝트는 NLP의 핵심 모델인 BERT (Bidirectional Encoder Representations from Transformers)의 전체 학습 파이프라인을 구현한 결과물입니다. 
밑바닥부터 모델을 학습시키는 사전 학습(Pretraining) 단계와, 특정 도메인 문제 해결을 위해 정품 모델을 활용하는 미세 조정(Fine-tuning) 단계를 모두 포함하고 있습니다.

## Deliverables
- Pretraining curves: 자체 구축한 BERT 모델의 사전 학습 진행 및 Loss 하락 로그 확보 (`metrics.json1`)
- Fine-tuning results: GLUE SST-2 데이터셋 기준 최종 검증 정확도(Accuracy) 91.86% 달성
---
### 1. Phase 1: Pretraining (`train_bert.py`)
> 목적: 위키백과 코퍼스(wiki_sample.txt)를 활용하여 백지상태의 BERT 모델에게 언어의 규칙을 가르치는 전반전.
- 주요 논문 구현 사항:
  - MLM (Masked Language Modeling) 및 NSP (Next Sentence Prediction) 학습 루프 구축.

### 2. Phase 2: Fine-Tuning (`finetune_glue.py`)
> 목적: 사전 학습이 완료된 구글 정품 모델을 가져와 영화 리뷰 데이터셋(GLUE SST-2)의 긍정/부정 뉘앙스를 분류하도록 훈련시키는 후반전.
- 주요 구현 사항:
  - Hugging Face `datasets` 및 `transformers` 라이브러리를 활용한 데이터 로드 및 토크나이징.
  - 파이토치 `DataLoader`를 통한 Batching 및 데이터 파이프라인(ETL) 설계.
  - 훈련 종료 후 별도의 검증(Evaluation) 루프를 통한 최종 모의고사 O/X 자동 채점 및 정확도(Accuracy) 산출.

---

## Final Results
- Task: 텍스트 감성 분류 (Sentiment Analysis)
- Dataset: GLUE SST-2 
- Validation Accuracy: `91.86%` (801/872 correct predictions)
  
# [Month 5]: LLM 4-bit Quantization & In-Context Learning

## 1. 프로젝트 목표
* 파라미터 수가 70억 개(7B)에 달하는 거대 언어 모델(LLM)을 제한된 환경(24GB VRAM)에서 구동하기 위한 메모리 최적화 기법 적용.
* 모델의 가중치를 업데이트하는 파인튜닝(Fine-tuning) 없이, 프롬프트 엔지니어링(0-shot, 1-shot, 5-shot)만으로 텍스트 분류(Sentiment Analysis) 태스크를 수행하는 In-Context Learning 성능 검증.
* 단벤치마크 데이터셋(GLUE SST-2)을 활용한 자동화 채점 루프를 구축하여 정량적인 성능 및 처리 속도(Throughput) 측정

## 2. 사용 환경 및 모델
* Model: `Mistral-7B-Instruct-v0.3` 
* Hardware: NVIDIA RTX 4090 (24GB VRAM)

## 3. 핵심 구현 내용 (Memory Optimization)
* BitsAndBytes 4-bit 양자화 (Quantization):
  * 거대한 16GB짜리 모델을 4090(24GB) GPU 메모리에 널널하게 올리기 위한 핵심 세팅 구현
  * 결과: 이론상 14~15GB가 필요한 모델을 3.86GB VRAM만 사용하여 로딩하는 데 성공

## 4. 정량적 분석 (GLUE SST-2 데이터셋 100개 샘플 검증)
Zero-shot Accuracy: 100.0% (100/100) / 처리 속도: 11.48 it/s
-> 1초에 반복문을 몇 번 돌았는지(iterations per second)를 자동으로 계산해 준 결과
-> 즉, 1초에 11.48개의 리뷰를 처리했다는 뜻

Few-shot (5-shot) Accuracy: 100.0% (100/100) / 처리 속도: 5.38 it/s
-> 즉, 1초에 5.38개의 리뷰를 처리했다는 뜻

분석: Mistral 7B 모델의 강력한 기본 성능으로 인해 두 환경 모두 100%의 정확도를 달성. 다만, 5-shot 적용 시 프롬프트 길이가 증가함에 따라 처리 속도가 절반 이하로 하락하는 비용적 트레이드오프(Trade-off)를 정량적으로 증명함. 

## 5. 핵심 인사이트 
1. 양자화의 위력: 성능 저하를 최소화하면서 VRAM 사용량을 획기적으로 줄이는 `BitsAndBytes`의 실무적 유용성 확인.
2. Causal LM의 출력 특성: 별도의 후처리(Slicing) 없이 원시 출력을 확인한 결과, 입력된 프롬프트를 앵무새처럼 반복한 뒤 예측값을 생성하는 자기회귀(Autoregressive) 특성을 직접 눈으로 검증.
3. In-Context Learning의 한계와 가능성: 파인튜닝 없이도 놀라운 추론 능력을 보여주었으나, 프롬프트가 길어질수록(5-shot) 모델의 출력을 완벽하게 통제하기 위해서는 더 정교한 시스템 프롬프트나 출력 후처리(Post-processing) 파이프라인이 필요함을 시사함.
4. 처리 비용(Throughput) 분석: 정확도를 높이기 위해 프롬프트 예시를 늘릴수록 연산량이 비례하여 증가하므로, 실무 도입 시 정확도(성능)와 처리 속도(비용)(Throughput) 간의 최적점을 찾는 것이 중요함.


# [Month 6 - langgraph-multi-agent-system]
# 주제 : "단일 GPU 환경에서의 Multi-agent 협업 전략 비교 분석: 프레임워크를 활용한 역할 분담 방식이 텍스트 추론 성능에 미치는 영향"
## Week 1: Single-Agent Pipeline Setup with LangGraph

## 1. 개요 (Overview)
본 주차의 목표는 본격적인 다중 에이전트(Multi-Agent) 시스템 구축에 앞서, 단일 GPU 환경(RTX 4090)에서 동작하는 1인 에이전트 파이프라인의 기초를 설계하는 것입니다. Month 5에서 검증한 4-bit 양자화 로컬 모델을 LangGraph 프레임워크의 노드(Node)로 이식하여 정상 작동 여부를 테스트했습니다.

## 2. 핵심 구현 내용 (Key Implementations)
1. 로컬 LLM 래핑 :
   4-bit로 로드된 Mistral 모델을 LangChain 생태계에서 사용할 수 있도록 텍스트 생성 전용 파이프라인을 Langchain 파이프라인에 딱 맞게 포장(wrapping)
2. 단일 노드 그래프 설계 (Graph Routing):
   `START` -> `agent` -> `END` 로 이어지는 단일 노드(Single-node) 워크플로우를 설계하고, 이를 실행 가능한 애플리케이션(`.compile()`)으로 빌드함.

## 4. 실행 결과 (Results)
* 테스트 쿼리: "What is the capital of France? Only answer with one word."
* 실행 결과: `🤖 에이전트 답변: Paris`
* 리소스 효율성: 4090 GPU의 24GB 중 단 3.86GB VRAM만을 점유하며 추론에 성공. 추후 2~3개의 에이전트를 동시 구동하기 위한 메모리 여유 공간을 완벽히 확보함.

## Week 2: Multi-Agent Collaboration Pipeline (Writer & Reviewer)

## 1. 개요 (Overview)
본 주차는 LLM 기반 다중 에이전트 시스템의 협업 전략(Collaboration Strategies)을 분석하기 위한 핵심 기반을 다지는 단계입니다. 단일 GPU(RTX 4090) 환경에서 4-bit 양자화된 로컬 모델을 활용하여, 서로 다른 역할을 부여받은 두 에이전트(Writer, Reviewer)가 데이터를 공유하고 협업하는 파이프라인을 LangGraph 프레임워크로 성공적으로 구현하였습니다.

## 2. 핵심 구현 내용 (Key Implementations)
1. 공유 상태(Shared State) 설계:
   * 두 에이전트가 공유할 공용메모장인 `AgentState`를 세분화하여 설계 (`topic`, `draft`, `review` 세 칸으로 구성).
2. 역할 기반 에이전트 (노드)(Role-playing Nodes) 정의:
   * Writer Agent: 주어진 주제(`topic`)에 대한 초기 초안(`draft`) 작성.
   * Reviewer Agent: 작성된 초안을 전달받아 검토하고 개선을 위한 비판적 피드백(`review`) 생성.
   * 단일 로컬 LLM(Mistral 7B)을 두 에이전트의 공통 추론 엔진(두뇌)으로 활용하여 24GB VRAM 내에서 메모리 효율성을 극대화함.
3. 멀티 에이전트 워크플로우(그래프) 라우팅 
   * START ➡️ `Writer` ➡️ `Reviewer` ➡️ END의 순차적 협업 파이프라인 구축 완료.

## 3. 실험 결과 (Results)
* 테스트 주제: "Artificial Intelligence in the future"
* 협업 결과:
  * [Writer 초안]: AI가 의료, 교육, 교통 등 다양한 분야에서 혁신을 주도할 것이라는 긍정적인 비전을 제시함.
  * [Reviewer 피드백]: 초안의 긍정적인 비전을 인정하면서도, 윤리적 고려사항(Ethical considerations)과 잠재적 위험성(Risks)에 대한 논의가 추가되어야 한다는 구체적이고 냉철한 피드백을 성공적으로 생성함.
* 의의: 두 에이전트가 단절되지 않고 하나의 상태(`State`)를 완벽하게 공유하며 상호작용(Interaction)을 수행하는 Multi-Agent 시스템의 기본 협업 아키텍처 검증을 완료함.

## Week 3: Multi-Agent Feedback Loop & Conditional Routing

## 1. 프로젝트 개요 (Overview)
LangGraph의 조건부 엣지(Conditional Edge)를 활용하여, 에이전트 간의 단순한 순차적 데이터 전달을 넘어 '평가 및 재작성(Cyclic Feedback Loop)'이 가능한 심화 Multi-Agent 협업 파이프라인을 구축함.

## 2. 핵심 구현 (Key Implementations)
.
* 조건부 라우팅 (Conditional Routing)
  * `conditional_router` 함수를 구현하여, Reviewer의 검토 결과(반복 횟수 조건)에 따라 데이터를 Writer에게 돌려보낼지(루프), 최종 승인할지(종료) 결정하는 분기 로직 적용.
* 피드백 기반 프롬프트 엔지니어링 (Feedback-Driven Prompting)
  * Writer 에이전트가 1회차(초안 작성)와 2회차 이상(피드백 반영 수정)일 때 각각 다른 프롬프트를 처리하도록 구축.

## 3. 실험 및 결과 (Results)

* 테스트 환경: 단일 RTX 4090 (24GB VRAM) / Mistral 7B 4-bit 양자화 모델 구동
* 수행 태스크: "Artificial Intelligence in the future" 주제의 소개글 작성 및 피드백 루프
* 실행 결과:
  * [Iteration 0] Writer 초안 생성 -> Reviewer의 윤리적 문제 추가 요청 (반려)
  * [Iteration 1] Writer 피드백 수용 및 내용 보강 -> Reviewer 최종 승인 (프로세스 정상 종료)
* 성과: 무한 루프 발생 없이, LLM 기반 에이전트들이 상호 작용을 통해 텍스트의 품질을 자율적으로 개선하는 Agentic Workflow의 작동을 성공적으로 증명함.


## Week 4: 3-Agent Collaboration System (Researcher, Writer, Reviewer)

## 1. 프로젝트 개요 (Overview)
기존의 2인(Writer, Reviewer) 피드백 루프 구조에 '자료 조사원(Researcher)' 에이전트를 도입하여, 팩트 기반의 정보 생성과 상호 검토가 결합된 3-Agent 워크플로우를 성공적으로 구축함. LLM의 고질적인 문제인 환각 현상(Hallucination)을 파이프라인 구조 레벨에서 방어하는 아키텍처를 설계함.

## 2. 핵심 구현 (Key Implementations)

* 상태 관리 확장 (Extended State Management)
  * `AgentState`에 `research_data` 필드를 추가하여, 수집된 팩트가 파이프라인 전체에 안전하게 공유되도록 설계함.
* 완벽한 역할 분담 및 프롬프트 엔지니어링 (Role Segregation)
  * [Researcher]: 주어진 주제에 대해 객관적인 핵심 팩트만 추출하여 제공 (RAG의 기초 역할 수행).
  * [Writer]: LLM의 자체 지식에 의존하지 않고, 오직 Researcher가 수집한 `research_data`만을 기반으로 초안 작성(Grounded Generation).
  * [Reviewer]: 작성된 초안의 완성도를 평가하고 피드백을 제공.
* 복합 그래프 라우팅 (Complex Graph Routing)
  * 단방향 순차 이동(`START` -> `Researcher` -> `Writer`)과 조건부 반복 루프(`Writer`  <-`Reviewer`)가 결합된 실무 수준의 LangGraph 토폴로지(Topology) 구현.

## 3. 실험 및 결과 (Results)

* 테스트 환경: 단일 RTX 4090 (24GB VRAM) / Mistral 7B (4-bit 양자화, NF4)
* 실행 결과:
  * 정보 조사 -> 초안 작성 -> 반려 및 수정 -> 최종 승인의 전 과정이 인간의 개입 없이완벽하게 수행됨을 검증 완료.
* 성과: 단일 에이전트의 한계를 넘어, 다수 에이전트 간의 '분업'과 '협업'을 통해 최종 산출물의 신뢰도와 품질을 자율적으로 극대화하는 시스템 구축.