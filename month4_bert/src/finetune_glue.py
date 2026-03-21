import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import os
import json
from datetime import datetime

def main():

    # 1. 훈련소 장치 세팅
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 모델 및 토크나이저 불러오기 (높은 점수라는 '결과물'을 확인하기 위해 허깅페이스의 정품 사전학습된 BERT 모델 사용)
    # 허깅페이스에서 기본형 bert tokenizer를 가져옴(uncased : 대소문자 무시. 전부 소문자로 바꿔서 숫자로 쪼갬) 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
    
    # 허깅페이스에서 기본형 bert model을 가져옴(num_labels = 2로 설정하여, 모델 머리 위에 '긍정/부정' 스위치 2개를 달아줌)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
    model.to(device) # 모델을 GPU 훈련장으로 보냄

    # 3. 데이터 다운로드 및 전처리 (텍스트 -> 숫자)
    print("GLEU SST-2 (영화 리뷰) 데이터셋을 다운로드")
    # train, validation, test 데이터가 각각 따로 분류되어있음
    dataset = load_dataset("glue", "sst2")

    # 전처리 함수 만들기
    def preprocess_function(examples):
        # 영화 리뷰(sentence)를 가져와서 128칸으로 자르고 빈칸(PAD)을 채움
        # 파이썬에서 객체 대입한 변수를 함수처럼 호출시 클래스 내부의 __call__함수 호출
        return tokenizer(examples["sentence"], truncation = True, padding = "max_length", max_length = 128)
    
    """
    python : map(적용할_함수_이름, 데이터_모음)
    데이터_모음의 요소들에 전부 앞에 함수를 적용해!
    huggingface : 데이터셋.map(함수, batched = True)
    데이터셋을 batch 단위로 꺼내서 함수를 통과시켜줘 
    """
    tokenized_datasets = dataset.map(preprocess_function, batched = True)   

    # 파이토치가 읽을 수 있는 텐서 형태로 바꿔줌.
    # tokenized_datasets에서 불필요한 글자 데이터는 치워버리고,
    # 모델이 학습하는데 필요한 핵심 숫자 데이터 3개만 남김
    # input_ids : 토크나이저가 글자를 숫자로 바꾼것
    # attention_mask : 빈칸을 쳐다보지 말라고 모델에게 알려주는 가이드라인
    # label : 정답지 (0: 부정, 1: 긍정)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns = ["input_ids", "attention_mask", "labels"]) 

    # 4. 데이터 로더 배치하기
    # 훈련용(train)는 섞어서 주고, 시험용(validation)은 안 섞고 그대로 줌
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle = True, batch_size = 16)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size = 16)


    # 5. 파인튜닝 위한 Optimizer 준비 
    optimizer = AdamW(model.parameters(), lr = 2e-5)
    epochs = 3
    print("본격적인 파인튜닝을 시작합니다!")

    # 6. 파인 튜닝 훈련 루프(Training Loop)
    for epoch in range(epochs):
        model.train() # 모델을 훈련 모드로 전환 (Dropout 등 활성화)
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            # 1. 포장된 배치(16개 데이터)를 통째로 GPU 훈련장으로 보냄
            batch = {k : v.to(device) for k, v in batch.items()}

            # 2. 모델에게 문제를 풀게 함 [순전파]
            outputs = model(**batch) # 키워드가변매개변수에 딕셔너리 전달

            # 3. 채점 (오답률 계산)
            loss = outputs.loss
            total_loss += loss.item()

            # 4. [역전파]
            loss.backward() # 역전파 : 어디서 틀렸는지 기울기 계산
            optimizer.step() # 옵티마지저가 계산된 기울기 방향으로 파라미터 실제 수정
            optimizer.zero_grad() # 다음 문제를 위해 계산된 기울기 초기화
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 훈련 완료! 평균 오답률 : {total_loss / len(train_dataloader):.4f}")

        # 5. 결과 기록
        log_dir = "runs/month4/bert_finetune"
        os.makedirs(log_dir, exist_ok = True)

        with open(f"{log_dir}/metrics.json1", "a") as f:
            # 각 에폭의 최종 평균 로스를 저장
            log_data = {
            "epoch" : epoch + 1,
            "avg_loss" : avg_loss,
            "timestamp" : str(datetime.now())

            }
            f.write(json.dumps(log_data) + "\n")
        print(f"학습 로그가 {log_dir}/metrics.json1에 저장되었습니다.")

    # 7. 모델 평가 (Evaluation Loop)
    print("파인튜닝된 모델을 테스트 시작")

    model.eval() # 모델을 평가 모드로 전환 (Dropout 등 비활성화)
    total_correct = 0 # 맞춘 문제 수를 저장할 공간
    total_samples = 0 # 전체 문제 수를 저장할 공간

    with torch.no_grad():
        
        for batch in eval_dataloader:
            # 1. 포장된 배치(16개 데이터)를 통째로 GPU 훈련장으로 보냄
            batch = {k : v.to(device) for k, v in batch.items()} 

            # 2. 모델에게 문제를 풀게 함 [순전파]
            outputs = model(**batch) # 키워드가변매개변수에 딕셔너리 전달
            
            """
            모델은 한 batch의 여러 리뷰 문제를 풀고, 각각의 리뷰에 대해 [부정점수, 긍정점수]를 
            요소로 하는 2차원 리스트 outputs을 리턴
            outputs.logits은 [[부정점수, 긍정점수],[부정점수, 긍정점수], ...] 이런 꼴임
            이걸 argmax()에 전달하면 각각의 리뷰에 대한 최댓값의 인덱스(0 or 1)을 1차원 리스트 꼴로 리턴
            예 : [1, 0, 1, 1, ...]
            """
            predictions = torch.argmax(outputs.logits, dim = -1) 


            """
            predictions와 labels를 나란히 놓고 비교해서
            같으면 True(1), 다르면 False(0)로 채운 1차원 리스트 리턴
            그 값들을 다 더하면 정답의 개수가 되고 알맹이만 뽑아내서 정답의 개수를
            correct에게 전달
            """

            correct = (predictions == batch["labels"]).sum().item() 
            total_correct += correct
            total_samples += len(predictions)

    # 최종 모델의 평가 결과(accuracy)를 발급
    accuracy = total_correct / total_samples
    print(f"최종 모델의 정확도(Accuracy) : {accuracy * 100:.2f}% ({total_correct}/{total_samples}문제 정답)")

            






# 테스트 실행 코드
if __name__ == "__main__":
    main()