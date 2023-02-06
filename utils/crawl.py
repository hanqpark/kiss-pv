import re
import sys
import time
import requests
import pandas as pd
import numpy as np
import urllib.request
import random
import numpy as np
import datetime
import networkx as nx
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import Counter
from itertools import combinations
from konlpy.tag import Kkma, Mecab
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


# 페이지 url 형식에 맞게 바꾸어 주는 함수 만들기
#입력된 수를 1, 11, 21, 31 ...만들어 주는 함수
def makePgNum(num):
    if num == 1:
        return num
    elif num == 0:
        return num+1
    else:
        return num+9*(num-1)


# 크롤링할 url 생성하는 함수 만들기(검색어, 크롤링 시작 페이지, 크롤링 종료 페이지)
#http://finance.naver.com/item/news_news.nhn?code={code}&page={page}&sm=title_entity_id.basic&clusterId=
def makeUrl(search, start_pg, end_pg, year, month):
    range_start = f"{year}{month}01"
    range_end = f"{year}{month}31"
    
    if start_pg == end_pg:
        start_page = makePgNum(start_pg)
        url=f"https://search.naver.com/search.naver?where=news&query={search}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={range_start}&de={range_end}&docid=&related=0&mynews=0&office_type=0&office_section_code=0&start={str(start_page)}&news_office_checked=&nso=so%3Ar%2Cp%3Afrom{range_start}to{range_end}&is_sug_officeid=0"
        #url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search}&ds={range_start}&de={range_end}&start={str(start_page)}&sort=1"
        print("생성url: ", url)
        return url
    else:
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = makePgNum(i)
            url=f"https://search.naver.com/search.naver?where=news&query={search}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={range_start}&de={range_end}&docid=&related=0&mynews=0&office_type=0&office_section_code=0&start={str(page)}&news_office_checked=&nso=so%3Ar%2Cp%3Afrom{range_start}to{range_end}&is_sug_officeid=0"

            #url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search}&ds={range_start}&de={range_end}&start={str(page)}1&sort=1"
            urls.append(url)
        print("생성url: ", urls)
        return urls    

# html에서 원하는 속성 추출하는 함수 만들기 (기사, 추출하려는 속성값)
def news_attrs_crawler(articles,attrs):
    attrs_content=[]
    for i in articles:
        attrs_content.append(i.attrs[attrs])
    return attrs_content


#html생성해서 기사크롤링하는 함수 만들기(url): 링크를 반환
def articles_crawler(i, url):
    #html 불러오기
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
    original_html = requests.get(i, headers=headers)
    html = BeautifulSoup(original_html.text, "html.parser")

    url_naver = html.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
    url = news_attrs_crawler(url_naver,'href')
    return url


#제목, 링크, 내용 1차원 리스트로 꺼내는 함수 생성
def makeList(newlist, content):
    for i in content:
        for j in i:
            newlist.append(j)
    return newlist


def crawl_news(name, year, month):
    mec = Mecab()
    # ConnectionError방지
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

    # 뉴스기사 쏠림 문제 해결위해 예외처리할 회사 리스트 (시총 기준 50위까지)
    ex_list=[
        '삼성전자',
        'LG에너지솔루션',
        'SK하이닉스',
        '삼성바이오로직스',
        'LG화학',
        '삼성SDI',
        '삼성전자우',
        '현대차',
        'NAVER',
        '카카오',
        '기아',
        'POSCO홀딩스',
        'KB금융',
        '셀트리온',
        '삼성물산',
        '신한지주',
        '현대모비스',
        '포스코케미칼',
        'LG전자',
        'SK이노베이션',
        'SK',
        '하나금융지주',
        '삼성생명',
        '카카오뱅크',
        'LG',
        '한국전력',
        'KT&amp;G',
        'LG생활건강',
        '고려아연',
        'HMM',
        '삼성전기',
        '두산에너빌리티',
        'SK텔레콤',
        '엔씨소프트',
        'S-Oil',
        '현대중공업',
        '에코프로비엠',
        '삼성에스디에스',
        '삼성화재',
        '우리금융지주',
        'KT',
        '셀트리온헬스케어',
        '대한항공',
        '크래프톤',
        '한화솔루션',
        '아모레퍼시픽',
        '기업은행',
        '카카오페이',
        '하이브',
        '엘앤에프'
    ]


    #####뉴스크롤링 시작#####
    # range_start = input("검색 시작할 기간을 입력해주세요:")
    # range_end = input("검색 종료할 기간을 입력해주세요:")



    #검색어 입력
    #예외처리리스트(시총50위)에 있으면 5pg-10pg에서 크롤링. 
    #아니면 1~5pg 범위로 크롤링  
    # search = input("검색할 키워드를 입력해주세요:")
    search = name
    if search in ex_list:
        print(search)
        page=5
        print("\n크롤링할 시작 페이지: ",page,"페이지")   

        page2=10
        print("\n크롤링할 종료 페이지: ",page2,"페이지")   

    else: 
        page = 1
        print("\n크롤링할 시작 페이지: ",page,"페이지")   
        #검색 종료할 페이지 입력
        page2 = 5
        print("\n크롤링할 종료 페이지: ",page2,"페이지")   
        
    if search in ex_list:
        search=search + " 하락"
        #검색 시작할 페이지 입력



    # naver url 생성
    url = makeUrl(search ,page, page2, year, month)

    #뉴스 크롤러 실행
    news_titles = []
    news_url =[]
    news_contents =[]
    news_dates = []
    for i in url:
        url = articles_crawler(i, url)
        news_url.append(url)

        
    #제목, 링크, 내용 담을 리스트 생성
    news_url_1 = []

    #1차원 리스트로 만들기(내용 제외)
    makeList(news_url_1,news_url)

    #NAVER 뉴스만 남기기
    final_urls = []
    for i in tqdm(range(len(news_url_1))):
        if "news.naver.com" in news_url_1[i]:
            final_urls.append(news_url_1[i])
        else:
            pass

    # 뉴스 내용 크롤링
    for i in tqdm(final_urls):
        #각 기사 html get하기
        news = requests.get(i,headers=headers)
        news_html = BeautifulSoup(news.text,"html.parser")

        # 뉴스 제목 가져오기
        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        if title == None:
            title = news_html.select_one("#content > div.end_ct > div > h2")
        
        # 뉴스 본문 가져오기
        content = news_html.select("div#dic_area")
        if content == []:
            content = news_html.select("#articeBody")

        # 기사 텍스트만 가져오기
        # list합치기
        content = ''.join(str(content))

        # html태그제거 및 텍스트 다듬기
        pattern1 = '<[^>]*>'
        title = re.sub(pattern=pattern1, repl='', string=str(title))
        content = re.sub(pattern=pattern1, repl='', string=content)
        pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
        content = content.replace(pattern2, '')

        news_titles.append(title)
        news_contents.append(content)

        try:
            html_date = news_html.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
            news_date = re.sub(pattern=pattern1,repl='',string=str(news_date))
        # 날짜 가져오기
        news_dates.append(news_date)

    print("검색된 기사 갯수: 총 ",(page2+1-page)*10,'개')
    print("\n[뉴스 제목]")
    print(news_titles)
    print("\n[뉴스 링크]")
    print(final_urls)
    print("\n[뉴스 내용]")
    print(news_contents)

    print('news_title: ',len(news_titles))
    print('news_url: ',len(final_urls))
    print('news_contents: ',len(news_contents))
    print('news_dates: ',len(news_dates))



    #데이터 프레임 만들기
    news_df = pd.DataFrame({'date':news_dates,'title':news_titles,'link':final_urls,'content':news_contents})

    #중복 행 지우기
    news_df = news_df.drop_duplicates(keep='first',ignore_index=True)
    print("중복 제거 후 행 개수: ",len(news_df))

    #데이터 프레임 저장
    #now = datetime.datetime.now(1) 
    #1news_df.to_csv('{}_{}.csv'.format(search,now.strftime('%Y%m%d')),encoding='utf-8-sig',index=False)


    content=''
    for i in range(0,19):
        content=content+news_df['content'][random.randrange(0,len(news_df['content']))]

    cleaned_content = re.sub('[^,.?!\w\s]','', content)  # 정규표현식 사용


    NN_words = [] 

    mec_pos = mec.pos(cleaned_content)
    for word, pos in mec_pos:
        if 'NN' in pos:
            NN_words.append(word)

    customized_stopwords = ['것', '등', '탓', '바', '용', '년', '개', '당', '면', '말','수','일','리','전','지','디','관','기자','배포','원','월','표','폐','앰','엔','비','티','씨','엠','앤','홀','엘','피','투','유','호','이','회사','한국','기업','에스','오전','오후','이날','너지','주가']

    unique_NN_words = set(NN_words)
    for word in unique_NN_words:
        if word in customized_stopwords or len(word)<=1:
            while word in NN_words: NN_words.remove(word)


    c = Counter(NN_words)
    print(c.most_common(30))   # 가장 빈번하게 나오는 n개의 단어 출력


    # 가장 많이 나오는 단어 20개 저장
    list_of_words = []
    for word, count in c.most_common(30):
        list_of_words.append(word)



    sentences = cleaned_content.split('.\n') 
    sentences1 = []
    sentences2 = []
    sentences3 = []
    for sentence in sentences:
        sentences1.extend(sentence.strip().split('. '))
    for sentence in sentences1:
        sentences2.extend(sentence.strip().split('!'))
    for sentence in sentences2:
        sentences3.extend(sentence.strip().split('?'))
    article_sentences = sentences3 

    print(article_sentences)


    ''' 가장 많이 출현하는 20개의 명사 단어들에 대해서 네트워크 생성하기 '''


    G = nx.Graph()
    G.add_nodes_from(list_of_words)   # node 생성 (가장 많았던 명사 단어 20개)

    print(G.nodes()) # nodes
    print(G.edges()) # edge, 즉 node 간의 관계는 아직 없는 상황


    for sentence in article_sentences:
        
        selected_words = []
        NN_words = [] 

        mec_pos = mec.pos(sentence)
        for word, pos in mec_pos:
            if 'NN' in pos:
                NN_words.append(word)
            
        for word in NN_words:
            if word in list_of_words:
                selected_words.append(word)

        selected_words = set(selected_words)

        for pair in list(combinations(list(selected_words), 2)):  
            if pair in G.edges(): 
                weight = G[pair[0]][pair[1]]['weight']
                weight += 1
                G[pair[0]][pair[1]]['weight'] = weight    
            else:
                G.add_edge(pair[0], pair[1], weight=1 )
            
    # 생성된 edge 확인해보기
    print(nx.get_edge_attributes(G, 'weight'))


    imp_list=[]
    imp_list=list(dict(G.degree).values())
    imp_node=sorted(imp_list.copy())[-6]

    print(imp_list) 


    ## 노드의 degree에 따라 color 다르게 설정하기
    color_map = []
    for node in G:
        if G.degree(node) >= imp_node: # 중요한 노드 (상위6개)
            color_map.append('pink') 
        else: 
            color_map.append('beige')    
        
        
    # font_location = 'utils/data/NanumBarunGothic.ttf'
    # font_name = font_manager.FontProperties(fname=font_location).get_name()
    rc('font', family='AppleGothic')
    plt.figure(figsize=(9, 7))
    plt.rcParams['font.family'] = 'AppleGothic'
    pos = nx.spring_layout(G)  # spring layout 사용

    ''' 첫번째 그림
    nx.draw_networkx(G, pos, node_color=color_map, edge_color='grey',node_size=500, font_family='AppleGothic')

    plt.axis('off') # turn off axis 
    plt.show() '''



    sizes=list(nx.degree_centrality(G).values())
    sizes_m=[]


    # 노드사이즈 조정
    sizes_m = [pow(s*10, 3.3) for s in sizes]


    ## 노드의 degree에 따라 color 다르게 설정하기
    color_map = []
    for node in G:
        if G.degree(node) >= imp_node: # 중요한 노드 (상위6개)
            #color_map.append('#FF7777') 
            color_map.append('#FF8000') 

        else:
            #color_map.append('#FF9933')   
            color_map.append('#FFB266')    

    plt.figure(figsize=(9, 7))
    pos = nx.spring_layout(G)  # spring layout 사용
    ##nods=list(nx.degree_centrality(G).values())
    #nodssize=map(lambda x: x*500,nods)
    #sizes=[list(nx.degree_centrality(G).values())*500 for node in G]
    #노드 크기는 단어별 중심도 기준으로 설정
    url = f'static/img/crawl/{name}_{year}-{month}.png'
    nx.draw_networkx(G, pos, node_color=color_map, edge_color='grey',node_size=sizes_m, font_family='AppleGothic')
    plt.axis('off') # turn off axis 
    plt.savefig(url, dpi=300)
    return url

if __name__ == "__main__":
    crawl_news("삼성전자")