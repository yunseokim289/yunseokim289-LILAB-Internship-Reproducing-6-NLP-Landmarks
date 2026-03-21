import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler # AMP 도구


# 직접 짠 모듈 볼러오기
from pretrain_dataset import BERTPretrainDataset
from modeling_bert import create_bert_model
from tokenizers import BertWordPieceTokenizer

def train():
    
    # 1. 장치 설정 (windows GPU는 CUDA, 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 준비 완료! 사용 장치: {device}")

    # 2. 토크나이저 불러오기
    tokenizer = BertWordPieceTokenizer("data/month4/tokenizer/vocab.txt", lowercase = True)
    
    # 짧은 문장은 빈칸([PAD])으로 512칸까지 꽉꽉 채우고(Padding)
    # 긴 문장은 512칸에서 가차 없이 자르도록(Truncation) 토크나이저에 룰을 추가
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(length=512)     

    # 3. 데이터 불러오기 (확보한 위키백과 데이터)
    print("위키백과 데이터를 읽어노는 중")
    with open("data/month4/wiki_sample.txt", "r", encoding = "utf-8") as f:
        # 텅 빈 줄은 무시하고 실제 텍스트로만 리스트를 만듦
        # 파이썬은 텅빈 것을 False로 취급
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 빠른 테스트(에러 확인)을 위해 우선 1,000줄만 잘라서 사용
    # 1000줄 자르기 삭제
    texts = lines



    # 4. 데이터 로더 장착 (컨베이어 벨트에 한 번에 8개씩(Batch) 올림)
    dataset = BERTPretrainDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)

    # 5. BERT 모델 뼈대 소환 후 GPU에 올리기
    model = create_bert_model().to(device)

    # 6. 최적화 도구 설정, [논문 Appendix A.2 (Pre-training Procedure)] 참고
    optimizer = AdamW(model.parameters(), lr = 1e-4)    

    # [AMP 추가] GPU 계산을 효율적으로 도와주는 스케일러 생성
    scaler = GradScaler()

    # 7. 본격적인 학습 시작
    epochs = 3
    model.train() # 모델에게 실전 학습 모드라고 알려줌

    for epoch in range(epochs):
        total_loss = 0

        for step, batch in enumerate(dataloader):
            # 문제지와 정답지를 GPU로 보냄
            # 객체[인덱스]시, dataset에 저장된 BERTPretrainDataset객체를 생성한
            # BERTPretrainDataset 클래스의 __getitem__ 함수 호출해서 해당 인덱스 값을 추출 후 리턴
           
            input_ids = batch["input_ids"][:, :512].to(device)
            labels = batch["labels"][:, :512].to(device)
            next_sentence_label = batch["next_sentence_label"].to(device)

            # 지난번 학습때 계산 기울기를 깨끗하게 지움
            optimizer.zero_grad()

            # [AMP 추가] 이 구간은 파이토치가 알아서 16비트로 초고속 계산
            # 모델에게 시험지를 풀게 하고 loss(손실)을 받음 [순전파]
            with autocast():
                outputs = model(
                input_ids = input_ids,
                labels = labels,
                next_sentence_label = next_sentence_label
                )
            
                # 객체 안의 여러 속성을 뽑아낼때는 객채.속성
                loss = outputs.loss 
            
            # [역전파]: 오답을 바탕으로 뇌세포 연결을 미세하게 수정
            # [AMP 추가] 역전파도 스케일러를 통해서 안전하게 진행
            scaler.scale(loss).backward() # 오차 역전파: 각 가중치가 틀린 만큼의 '기울기' 계산
            scaler.step(optimizer) # 가중치 업데이트: 기울기 방향으로 모델 수정
            scaler.update()
            
            total_loss += loss.item()
        
        # len(객체)시 객체를 생성한 클래스의 __len__함수를 호출 
        # 총 batch의 개수 리턴
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 완료! 평균 오답률(Loss): {avg_loss :.4f} ")


        # 8. 결과 기록
        log_dir = "runs/month4/bert_pretrain_mlm_25"
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


# 테스트 실행 코드
if __name__ == "__main__":
    train()