import json
# huggingface에서 제공하는 BERT 전용 config, 사전 학습용 BERT모델 뼈대를 가져옴
from transformers import BertConfig, BertForPreTraining

def create_bert_model(config_path = "configs/month4/month4_bert_config.json"):
    print("Config를 바탕으로 BERT모델 뼈대를 조립")

    with open(config_path, "r", encoding = "utf-8") as f:
        config_dict = json.load(f)
    
    # json 딕셔너리를 huggingface가 알아들을 수 있는 'BertConfig' 객체로 변환
    config = BertConfig(
        vocab_size = config_dict["vocab_size"],
        hidden_size = config_dict["hidden_size"],
        num_hidden_layers = config_dict["num_hidden_layers"],
        num_attention_heads = config_dict["num_attention_heads"],
        max_position_embeddings = config_dict["max_position_embeddings"],
        # 아래 두개는 업계 표준 프랜스포머 내부의 정교한 세팅
        intermediate_size = config_dict["hidden_size"] * 4, # FFN의 확장된 차원의 크기
        type_vocab_size = 2 # NSP 훈련을 위해 문장 A와 B를 구분하는 0과 1 (두가지 종류)
    )

    # 설계도(config)를 바탕으로 실제 BERT모델을 생성
    # BertForTraining : 기본 BERT 몸통 위에 MLM(빈칸 맞추기) 머리와 NSP(다음 문장 맞추기) 머리가  
    # 모두 달려있는 사전 학습용 모델
    model = BertForPreTraining(config)     
    
    return model


if __name__ == "__main__":
    bert_model = create_bert_model()
    
    print("\n✨ BERT 모델 뼈대 조립 완료!")
    # 모델 안에 학습해야 할 가중치(파라미터)가 총 몇 개인지 계산해서 보여줍니다.
    print(f"총 파라미터(가중치) 개수: {bert_model.num_parameters():,}개")