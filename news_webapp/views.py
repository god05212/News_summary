import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import requests
from bs4 import BeautifulSoup
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from django.shortcuts import render
from collections import Counter
from konlpy.tag import Okt

# 토크나이저와 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

# 불용어 리스트
stopwords = [
    '의', '가', '이', '는', '다', '을', '를', '에', '에서', '에게', '과', '와',
    '도', '로', '저', '이것', '그것', '저것', '이런', '저런', '어떤',
    '무엇', '어디', '있다', '없다', '하다', '되다', '아니다', '보이다', '오다',
    '가다', '나가다', '들어가다', '된다', '한다', '해야', '해', '하고', '하면서',
    '그러나', '그런', '그래서', '즉', '바로', '등', '이외', '수', '경우', '관련',
    '내용', '일', '각', '일부', '그중', '그 외', '측', '대', '기간', '상태',
    '상황', '요인', '기준', '각종', '변동', '방법', '경향', '그들', '우리', '너',
    '저기', '이곳', '저곳', '그곳', '여기', '저기', '데', '및', '같은', '모든',
    '것', '들', '나', '우리는', '있는', '있는지', '한다면', '되는지', '되고',
    '되었던', '되고자', '해야', '하면서', '할지', '하고', '해', '통해'
]

def first_view(request):
    # POST 요청이 들어오면 쿼리 처리
    if request.method == 'POST':
        query = request.POST.get('query')
        return news_summary(request, query)
    return render(request, 'news_webapp/first_view.html', {})

def summarize_texts(texts):
    # 여러 텍스트 요약 생성
    summaries = []
    for text in texts:
        text = text.replace('\n', ' ')
        raw_input_ids = tokenizer.encode(text, return_tensors='pt')
        input_ids = torch.cat((torch.tensor([[tokenizer.bos_token_id]]),
                                raw_input_ids,
                                torch.tensor([[tokenizer.eos_token_id]])), dim=1)

        summary_ids = model.generate(input_ids, num_beams=4, max_length=512, eos_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        summaries.append(result)
    return summaries

def extract_nouns(text):
    # 명사 추출 및 빈도수 계산
    okt = Okt()
    nouns = okt.nouns(text)
    filtered_nouns = [word for word in nouns if word not in stopwords]

    # 길이가 1보다 큰 경우만 남김
    filtered_nouns = [word for word in filtered_nouns if len(word) > 1]

    return Counter(filtered_nouns)  # Counter 객체 반환

def news_summary(request, query):
    num_articles = 15
    news_articles = []
    start = 1

    while len(news_articles) < num_articles:
        url = f"https://search.naver.com/search.naver?where=news&query={query}&start={start}"

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"뉴스 기사를 가져오는 중 오류 발생: {e}")
            break

        soup = BeautifulSoup(response.content, "html.parser")
        articles_on_page = []

        for news in soup.find_all("div", class_="news_area"):
            title = news.find("a", class_="news_tit").text.strip()
            link = news.find("a", class_="news_tit")["href"]
            text = news.find("div", class_="news_dsc").text.strip() if news.find("div", class_="news_dsc") else "본문 없음"
            articles_on_page.append((title, link, text))

        texts = [article[2] for article in articles_on_page]
        summarized_texts = summarize_texts(texts)

        for i, article in enumerate(articles_on_page):
            news_articles.append((article[0], article[1], summarized_texts[i]))

        if len(news_articles) >= num_articles:
            break

        start += 10

    all_summaries = ' '.join([summary for _, _, summary in news_articles])
    noun_counts = extract_nouns(all_summaries)  # 명사와 빈도수 추출

    # 가장 빈도수가 높은 10개의 명사 선택
    top_nouns = noun_counts.most_common(10)

    return render(request, 'news_webapp/news_summary.html', {
        'query': query,
        'articles': news_articles[:num_articles],
        'top_nouns': top_nouns,  # 상위 10개 명사 전달
    })
