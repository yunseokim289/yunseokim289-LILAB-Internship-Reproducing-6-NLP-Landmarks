# huggingface의 빠르고 강력한 tokenizers 라이브러리를 사용
from tokenizers import BertWordPieceTokenizer
import os
import json

# [Devlin et al. 2018, session 3] 
# We use WordPiece embeddings (Wu et al.,2016) with a 30,000 token vocabulary. 
# BERT의 핵심인 WordPiece embedding(=BERT tokenizer : 텍스트를 잘게 쪼개는 기계(함수)
# 를 학습시키는 코드

def train_bert_tokenizer():
    print("WordPiece 토크나이저 학습을 시작")

    config_path = "configs/month4/month4_bert_config.json"
    with open(config_path, "r", encoding = "utf-8") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"] # 모델이 알고 있는 단어의 총 개수 : 30000

    # [session 3] WordPiece embedding 생성
    tokenizer = BertWordPieceTokenizer(
        clean_text = True, # 텍스트 청소
        handle_chinese_chars = True, # 한자 주변에 강제로 띄어쓰기를 넣어서 분리
        strip_accents = True, # 악센트 제거
        lowercase = True # 대소문자 구분 x, 소문자로 통일
    )   

    # prepare_data.py에서 다운바았던 위키백과 데이터의 경로
    data_files = ["data/month4/wiki_sample.txt"]
    
    # tokenizer에게 데이터를 주고, 3만 개 짜리 단어장을 구축하도록 훈련시킴
    tokenizer.train(
        files = data_files,
        vocab_size = vocab_size,
        min_frequency = 2, # 최소 2번 이상 등장한 단어만 취급하여 퀄리티를 높임
        # BERT 모델이 작동하기 위해 반드시 필요한 5개의 특수 토큰
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # 학습된 단어장을 저장할 폴더 생성
    save_dir = "data/month4/tokenizer"
    os.makedirs(save_dir, exist_ok = True)
    
    # tokenizer에게 네 머릿속에 있는 그 단어들을 아까 만든 폴더(save_dir) 안에 파일로 저장해!
    # 컴퓨터가 알아서 해당 폴더 안에 vocab.txt라는 이름의 텍스트 파일(단어장)을 딱 만들어냅니다.
    tokenizer.save_model(save_dir) 
    
    print(f"토크나이저 학습 완료! 단어 30,000개가 '{save_dir}/vocab.txt' 단어장에 채워졌습니다.")
    

# 테스트 실행코드    
if __name__ == "__main__":
    train_bert_tokenizer()    