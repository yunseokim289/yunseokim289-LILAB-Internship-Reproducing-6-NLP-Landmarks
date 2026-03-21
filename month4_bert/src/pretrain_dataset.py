import torch
from torch.utils.data import Dataset
import random
import json
from tokenizers import BertWordPieceTokenizer

# [Devlin et al. 2018, session 3.3] 
# Task 1: MLM (빈칸 뚫기)
# Task 2: NSP (문장 순서 맞추기

class BERTPretrainDataset(Dataset):
    def __init__(self, texts, tokenizer, config_path = "configs/month4/month4_bert_config.json"):
        self.texts = texts
        self.tokenizer = tokenizer

        with open(config_path, "r", encoding = "utf-8") as f:
            config = json.load(f)

        # Ablation 조건 3가지 세팅
        self.max_len = config["max_position_embeddings"]  # 최대 길이 (128)
        self.mlm_prob = config["mlm_probability"] # 빈칸 뚫을 확률 (첫 번째 비교 실험 : 0.15-> 0.25)
        self.use_nsp = config["use_nsp"] # NSP 훈련 스위치 


    def __len__(self):
        return len(self.texts) # 데이터셋의 총 문장의 개수를 리턴

    def __getitem__(self, idx):

        # 1. 기준이 되는 첫 번째 문장(Sentence A)을 가져옴
        text_a = self.texts[idx]

        # 2. Task 2 : NSP (문장 순서 맞추기)    
        is_next = 1
        text_b = None

        # configs에서 use_nsp 스위치를 켰을 때만 작동시킴
        if self.use_nsp:
            # random.random()는 0~1 사이의 난수를 뽑음 (50% 확률 만들기)
            if random.random() < 0.5 and idx + 1 < len(self.texts):
               # 50% 확률로 진짜 다음 문장을 가져와서 붙임  (IsNext = 1)
               text_b = self.texts[idx + 1]

            else:
                # 나머지 50% 확률로 엉뚱한 무작위 문장을 붙임 (NotNext = 0)
                random_idx = random.randint(0, len(self.texts) - 1)
                text_b = self.texts[random_idx]
                is_next = 0  


        # tokenizer을 사용해 문장을 쪼개고 숫자로 바꿈(인코딩)
        # text_b가 있으면 알아서 중간에 [SEP]을 넣어줌

        if text_b:
            encoded = self.tokenizer.encode(text_a, text_b)
        
        else:
            encoded = self.tokenizer.encode(text_a)

        input_ids = encoded.ids[:self.max_len] # 길이를 max_len(128)에 맞게 싹둑 자름
        
        # 만약 input_ids가 128보다 짧으면 남는 공간을 0번 토큰("[PAD]")으로 채움
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [0] # pad_len
        
       
        # 3. Task 1 : MLM (빈칸 뚫기)
        # 원본 숫자들을 복사해서 정답지(LABELS)를 따로 만듦
        input_ids = torch.tensor(input_ids)
        labels = input_ids.clone()

        # 특수 토큰들 (0:PAD, 1:UNK, 2:CLS. 3:SEP, 4:MASK)
        special_tokens = [0, 1, 2, 3, 4]

        # 각 단어(토큰)마다 돌아가면서 검사
        for i, token_id in enumerate(input_ids):
            # 일반 단어일때만 (특수 토큰은 건드리지 않음)
            if token_id.item() not in special_tokens:
                # mlm_prob(예: 15%) 확률에 당첨되면?
                if random.random() < self.mlm_prob:
                    # i번째 자리를 마스킹하고 MASK 토큰의 번호인 4번을 덮어씌움
                    input_ids[i] = 4
                else:
                    # 마스킹되지 않고 살아남은 단어는 정답지에서 무시(-100으로 덮어씌움)
                    labels[i] = -100     

            else:
                # 특수 토큳들도 정답에서 무시
                labels[i] = -100

        return {
            "input_ids": input_ids, # 빈칸 뚫어놓은 문제지
            "labels": labels, # 빈칸 자리에만 원래 단어가 들어있는 정답지
            "next_sentence_label": torch.tensor(is_next) # 두 문장이 이어지는지 맞추는 정답지 (1 또는 0)

        }                


"""
# 테스트 실행 코드
if __name__ == "__main__":
    # 방금 만든 3만 개짜리 단어장을 불러옵니다.
    tokenizer = BertWordPieceTokenizer("data/month4/tokenizer/vocab.txt", lowercase=True)
    
    # 임시로 문장 3개를 만들어서 테스트해 봅니다.
    sample_texts = [
        "Hello, my name is Yoonseo.",
        "I am learning deep learning.",
        "Today's lunch was very delicious."
    ]
    
    # 데이터셋을 만듭니다.
    dataset = BERTPretrainDataset(texts=sample_texts, tokenizer=tokenizer)
    
    # 첫 번째(0번) 데이터를 뽑아봅니다.
    sample_data = dataset[0]
    
    print("✅ 데이터 로더 테스트 성공!")
    print(f"입력 길이 (input_ids): {sample_data['input_ids'].shape}")
    print(f"정답 길이 (labels): {sample_data['labels'].shape}")
    print(f"NSP 정답 (next_sentence_label): {sample_data['next_sentence_label']}")\

"""