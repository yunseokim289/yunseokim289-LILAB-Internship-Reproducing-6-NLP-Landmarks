from datasets import load_dataset
import os

def download_and_save_wiki():
    print("wikipedia 데이터 다운로드 시작 (지침서 10GB 이하 조건을 위해 1%만 샘플링)")
    
    # load_dataset('데이터셋_이름', '버전', split = '가져올_비율')
    dataset = load_dataset('wikimedia/wikipedia', '20231101.en',split = 'train[:1%]')
    
    save_path = 'data/month4/wiki_sample.txt'
    
    with open(save_path, 'w', encoding = 'utf-8') as f:
        
        for item in dataset:
            
            # data에서 본문 텍스트만 뽑아내서 엔터키를 전부 띄어쓰기로 바꿈
            # BERT의 입력으로 넣기 좋게 한 줄로 길게 연결하는 전처리 과정
            text = item['text'].replace('\n', ' ') 

            f.write(text + '\n')

    print(f"데이터 저장 완료! 저장 위치 : {save_path}")


# 테스트 실행 코드
if __name__ == "__main__":
    download_and_save_wiki()

