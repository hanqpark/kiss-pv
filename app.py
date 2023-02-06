import os.path
import pandas as pd
from datetime import datetime
from collections import defaultdict
from flask import Flask, session, render_template, redirect, request, url_for 
from utils.rebalancing import rebalance
from utils.quant import quanting
from utils.crawl import crawl_news


def get_etf_start_date(name):
    idx = ke_df[ke_df['name']==name].index
    return ke_df['start_date'][idx].values[0]

def get_etf_ticker(name):
    idx = ke_df[ke_df['name']==name].index
    return ke_df['ticker'][idx].values[0]

def get_start_date(name):
    idx = ks_df[ks_df['name']==name].index
    return ks_df['start_date'][idx].values[0]

def get_ticker(name):
    idx = ks_df[ks_df['name']==name].index
    return ks_df['ticker'][idx].values[0]

def get_finance_data(path):
    #  [코드 4.6] 재무 데이터 전처리하는 함수 (CH4. 전략 구현하기.ipynb)
    data_path = path
    raw_data = pd.read_excel(data_path)
    raw_data=raw_data.set_index('Unnamed: 0')
    big_col = list(raw_data.columns)
    small_col = list(raw_data.iloc[0])
    
    new_big_col = []
    for num, col in enumerate(big_col):
        if 'Unnamed' in col:
            new_big_col.append(new_big_col[num-1])
        else:
            new_big_col.append(big_col[num])
            
    raw_data.columns = [new_big_col, small_col]
    clean_df = raw_data.loc[ raw_data.index.dropna() ]
    return clean_df

def data_postprocess(info, etf=False):
    daily_df, monthly_df, annual_df, total_balance, total_invest_money, metrics = rebalance(info, etf=etf)
    annual_df = round(annual_df*100, 2)
    monthly_df = round(monthly_df*100, 2)
    
    # 일별 정보
    d_label = list(t.strftime("%Y-%m-%d") for t in list(daily_df.index))
    d_port = list(daily_df['backtest'])
    
    # 월별 정보
    m_label = list(t.strftime("%Y-%m") for t in list(monthly_df.index))
    month_dict = defaultdict()
    month_dict['연월'] = m_label
    month_dict['포트폴리오'] = list(monthly_df['backtest'])
    for t in info['ticker']:
        month_dict[t] = list(monthly_df[t])
    
    # 연별 정보
    y_label = list(t.strftime("%Y") for t in list(annual_df.index))
    year_dict = defaultdict()
    year_dict['연도'] = y_label
    year_dict['포트폴리오'] = list(annual_df['backtest'])
    for t in info['ticker']:
        year_dict[t] = list(annual_df[t])
    
    #포트폴리오 개요
    try:
        outline = defaultdict(
            ticker = list(get_ticker(name) for name in info['ticker']),
            name = info['ticker'],
            start_date = list(get_start_date(name) for name in info['ticker']),
            ratio = info['ratio']
        )
    except IndexError:
        outline = defaultdict(
            ticker = list(get_etf_ticker(name) for name in info['ticker']),
            name = info['ticker'],
            start_date = list(get_etf_start_date(name) for name in info['ticker']),
            ratio = info['ratio']
        )
        
    # 포트폴리오 성과 요약
    annual_result = list(annual_df['backtest'])
    summary = [
        f"{int(info['moneyToStart'][0])}만원",
        f"{int(info['monthlySave'][0])}만원",
        f"{int(total_invest_money)//10000}만원",
        f"{format(int(total_balance), ',')}원",
        f"{max(annual_result)}%",
        f"{min(annual_result)}%",
        metrics['연평균성장률'],
        metrics['최대 손실 낙폭'],
        metrics['샤프 비율']
    ]
    return d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics

application = Flask(__name__)

@application.route("/")
def main():
    return render_template("index.html")


@application.route("/quant", methods=['GET', 'POST'])
def quant():
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict()
        for k, v in request.form.lists():
            info[k] = v[0]
        daily_df, monthly_df, annual_df, metrics = quanting(info, fs_df, fr_df, invest_df, price_df)
        
        # 일별 정보
        d_label = list(t.strftime("%Y-%m-%d") for t in list(daily_df.index))
        d_label.pop(0)
        d_port = list(daily_df['총변화율']+1)
        
        # 월별 정보
        m_label = list(t.strftime("%Y-%m") for t in list(monthly_df.index))
        m_df = list(monthly_df.dropna()+1)
        
        # 연별 정보
        y_label = list(t.strftime("%Y") for t in list(annual_df.index))
        y_label.pop(0)
        y_df = list(annual_df.dropna()+1)
        
        # 포트폴리오 성과 요약
        summary = [
            f"{info['moneyToStart']}만원",
            f"{format(int(daily_df['주식포트폴리오'][-1]), ',')}원",
            f"{format(int(daily_df['현금포트폴리오'][-1]), ',')}원",
            f"{format(int(daily_df['종합포트폴리오'][-1]), ',')}원",
            f"{round(max(y_df), 2)}%",
            f"{round(min(y_df), 2)}%",
            metrics['연평균성장률'],
            metrics['최대 손실 낙폭'],
            metrics['샤프 비율']
        ]
        
        return render_template(
            "backtesting/quant.html", 
            daily_port=d_port, 
            daily_label=d_label,
            monthly_port=m_df,
            monthly_label=m_label,
            annual_port=y_df,
            annual_label=y_label,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template("backtesting/quant.html")


@application.route("/rebalancing-korea", methods=['GET', 'POST'])
def rebalancing_korea():
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict(list)
        for k, v in request.form.lists():
            info[k] = v
        d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics = data_postprocess(info)
        return render_template(
            "backtesting/rebalancing-korea.html",
            daily_port=d_port, 
            daily_label=d_label,
            monthly_port=month_dict,
            monthly_label=m_label,
            annual_port=year_dict,
            annual_label=y_label,
            outline=outline,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template("backtesting/rebalancing-korea.html")


@application.route("/rebalancing-usa", methods=['GET', 'POST'])
def rebalancing_usa():
    if request.method == "POST":
        info = dict(request.form)
        return render_template("backtesting/rebalancing-usa.html", info=info)
    else:
        return render_template("utilities/blank.html")

'''
@application.route("/portfolio-guru")
def guru():
    return render_template("portfolio/guru.html")
'''

@application.route("/ray-dalio", methods=['GET', 'POST'])
def ray():
    portfolio={
                'KODEX KRX300': 25,
                'ACE 중장기국공채액티브': 40,
                'TIGER 국채3년': 15,
                'KODEX WTI원유선물(H)': 7.5,
                'KODEX 골드선물(H)': 7.5
            }
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict(list)
        for k, v in request.form.lists():
            info[k] = v
        info['name'] = portfolio.keys()
        info['ratio'] = portfolio.values()
        d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics = data_postprocess(info, etf=True)
        return render_template(
            "backtesting/rebalancing-guru.html",
            title="레이 달리오's 올웨더 포트폴리오",
            portfolio=portfolio,
            daily_port=d_port,
            daily_label=d_label,
            monthly_port=month_dict,
            monthly_label=m_label,
            annual_port=year_dict,
            annual_label=y_label,
            outline=outline,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template(
            "backtesting/rebalancing-guru.html",
            # "utilities/blank.html",
            title="레이 달리오's 올웨더 포트폴리오",
            portfolio=portfolio
        )


@application.route("/harry-browne", methods=['GET', 'POST'])
def harry():
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict(list)
        for k, v in request.form.lists():
            info[k] = v
        d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics = data_postprocess(info)
        return render_template(
            "backtesting/rebalancing-korea.html",
            title="해리 브라운's 영구 포트폴리오",
            daily_port=d_port, 
            daily_label=d_label,
            monthly_port=month_dict,
            monthly_label=m_label,
            annual_port=year_dict,
            annual_label=y_label,
            outline=outline,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template("utilities/blank.html",)


@application.route("/sixty-forty", methods=['GET', 'POST'])
def sixty_forty():
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict(list)
        for k, v in request.form.lists():
            info[k] = v
        d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics = data_postprocess(info)
        return render_template(
            "backtesting/rebalancing-korea.html",
            title="주식·채권 60/40",
            daily_port=d_port, 
            daily_label=d_label,
            monthly_port=month_dict,
            monthly_label=m_label,
            annual_port=year_dict,
            annual_label=y_label,
            outline=outline,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template("utilities/blank.html",)


@application.route("/forty-sixty", methods=['GET', 'POST'])
def forty_sixty():
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict(list)
        for k, v in request.form.lists():
            info[k] = v
        
        d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics = data_postprocess(info)
        return render_template(
            "backtesting/rebalancing-korea.html",
            title="주식·채권 40/60",
            daily_port=d_port, 
            daily_label=d_label,
            monthly_port=month_dict,
            monthly_label=m_label,
            annual_port=year_dict,
            annual_label=y_label,
            outline=outline,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template("utilities/blank.html",)


@application.route("/portfolio-kis", methods=['GET', 'POST'])
def kis():
    companies = ['팬오션', 'NAVER', '포스코케미칼', '삼성엔지니어링', 'HL만도', 'POSCO홀딩스', '엔씨소프트', '삼성바이오로직스', 'BGF리테일']
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict(list)
        for k, v in request.form.lists():
            info[k] = v
        info['ticker'] = companies
        d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics = data_postprocess(info)
        return render_template(
            "backtesting/rebalancing-sec.html",
            title="한국투자증권 추천 종목",
            daily_port=d_port, 
            daily_label=d_label,
            monthly_port=month_dict,
            monthly_label=m_label,
            annual_port=year_dict,
            annual_label=y_label,
            outline=outline,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template(
            "backtesting/rebalancing-sec.html",
            title="한국투자증권 추천 종목",
            companies=companies
        )


@application.route("/portfolio-kb", methods=['GET', 'POST'])
def kb():
    companies=['LG전자', 'S-Oil', 'NAVER', 'SK하이닉스', '삼성전자', '삼성생명', '하나금융지주', '현대모비스', '현대글로비스', '삼성SDI']
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict(list)
        for k, v in request.form.lists():
            info[k] = v
        info['ticker'] = companies
        d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics = data_postprocess(info)
        return render_template(
            "backtesting/rebalancing-sec.html",
            title="KB증권 추천 종목",
            daily_port=d_port, 
            daily_label=d_label,
            monthly_port=month_dict,
            monthly_label=m_label,
            annual_port=year_dict,
            annual_label=y_label,
            outline=outline,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template(
            "backtesting/rebalancing-sec.html",
            title="KB증권 추천 종목",
            companies=companies
        )


@application.route("/portfolio-samsung", methods=['GET', 'POST'])
def samsung():
    companies = ['HMM', 'LG전자', 'NAVER', 'POSCO홀딩스', '삼성전기', '솔루엠', '아모레G', '엔씨소프트', '이오테크닉스', '호텔신라']
    if request.method == "POST":
        # json 형식의 input value 생성
        info = defaultdict(list)
        for k, v in request.form.lists():
            info[k] = v
        info['ticker'] = companies
        d_port, d_label, month_dict, m_label, year_dict, y_label, outline, summary, metrics = data_postprocess(info)
        return render_template(
            "backtesting/rebalancing-sec.html",
            title="삼성증권 추천 종목",
            daily_port=d_port, 
            daily_label=d_label,
            monthly_port=month_dict,
            monthly_label=m_label,
            annual_port=year_dict,
            annual_label=y_label,
            outline=outline,
            summary=summary,
            metrics=metrics
        )
    else:
        return render_template(
            "backtesting/rebalancing-sec.html",
            title="삼성증권 추천 종목",
            companies = ['HMM', 'LG전자', 'NAVER', 'POSCO홀딩스', '삼성전기', '솔루엠', '아모레G', '엔씨소프트', '이오테크닉스', '호텔신라']
        )
        
@application.route("/static/img/crawl/<img>.png")
def baek(img):
    company, date = img.split('_')
    year, month = date.split('-')
    crawl_news(company, year, month)
    return render_template('backtesting/rebalancing-korea.html')


@application.route("/metrics")
def metrics():
    return render_template("utilities/metrics.html")


@application.route("/support")
def support():
    return render_template("utilities/support.html")


@application.route("/search", methods=['GET'])
def search():
    if request.method == 'GET':
        name = request.args.get('company')
        img = f'static/img/crawl/{name}_{year}-{month}.png' if os.path.isfile(f'static/img/crawl/{name}_{year}-{month}.png') else crawl_news(name)
        ticker = get_ticker(name)
        return render_template("utilities/search.html", name=name, ticker=ticker, url=img)
    

@application.route("/popup")
def popup():
    return render_template("users/user-profile.html")
    
    
@application.route("/profile")
def profile():
    return render_template("users/user-profile.html")


@application.route("/holdings")
def holdings():
    return render_template("users/user-holdings.html")


@application.route("/login")
def login():
    return render_template("users/before-login.html")


@application.route("/register")
def register():
    return render_template("users/before-register.html")


@application.route("/profile/modify")
def user_modify():
    return render_template("users/user-modify.html")


@application.route("/forgot-password")
def forgot_password():
    return render_template("users/before-password.html")


@application.route("/404")
def page_not_found():
    return render_template("utilities/404.html")


@application.route("/blank")
def blank():
    return render_template("utilities/blank.html")


@application.route("/reference")
def reference():
    return render_template("reference.html")


@application.errorhandler(404)
def page_not_found(error):
    return render_template('utilities/404.html')


@application.errorhandler(500)
def page_not_found(error):
    return render_template('utilities/404.html')



if __name__ == "__main__":
    # 리밸런싱
    ks_df = pd.read_csv('utils/data/korea_stock.csv')
    
    # ETF
    # ke_df = pd.read_csv('utils/data/korea_etf.csv')
    
    # 퀀트
    # fs_path = 'utils/data/재무제표데이터.xlsx'
    # fs_df = get_finance_data(fs_path)
    # fr_path = 'utils/data/재무비율데이터.xlsx'
    # fr_df = get_finance_data(fr_path)
    # invest_path = 'utils/data/투자지표데이터.xlsx'
    # invest_df = get_finance_data(invest_path)
    # price_path = 'utils/data/가격데이터.csv'
    # price_df = pd.read_csv(price_path)
    # price_df = price_df.set_index('Unnamed: 0')
    # price_df.index = [price_df.index[i][:10] for i in range(len(price_df['060310']))]
    # price_df.index = pd.to_datetime(price_df.index)
    application.run(host='0.0.0.0', port="8080")