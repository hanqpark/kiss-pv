import pprint
import requests
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from lxml import html
from scipy.stats import norm
from bs4 import BeautifulSoup
from datetime import datetime
from itertools import groupby, chain
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote


class Core():
    def __init__(self):
        self.annual=252
    # 산술평균 수익률
    def average(self, returns):
        return returns.mean()*self.annual
    # 기하평균 수익률
    def cagr(self, returns):
        return (1+returns).prod() ** (self.annual/len(returns))-1
    # 표준편차
    def stdev(self, returns):
        return returns.std() * np.sqrt(self.annual)
    # 하방 표준편차
    def downdev(self, returns, target=0.0):
        returns = returns.copy()
        returns.loc[returns>target]=0
        summation = (returns ** 2).sum()
        return np.sqrt(self.annual * summation / len(returns))
    # 상방 표준편차
    def updev(self, returns, target=0.0):
        returns = returns.copy()
        returns.loc[returns < target] = 0
        summation = (returns ** 2).sum()
        return np.sqrt(self.annual * summation / len(returns))
    # 공분산
    def covar(self, returns, benchmark):
        return returns.cov(benchmark) * self.annual
    # 상관계수
    def correl(self, returns, benchmark):
        return returns.corr(benchmark)
    # 베타
    def beta(self, returns, benchmark):
        return returns.cov(benchmark) / returns.std() ** 2
    # 알파 : R펀드실제 －[R무위험 ＋ β펀드 {E(R시장) －R무위험 }]
    def alpha(self, returns, benchmark):
        return (1+returns).prod() - (r_f + (returns.cov(benchmark)/returns.std() ** 2)) * (benchmark.mean() - r_f)
        
    # 한번에 출력
    def print_result(self, returns, benchmark, target=0.0):
        average = self.average(returns)
        cagr = self.cagr(returns)
        stdev = self.stdev(returns)
        downdev = self.downdev(returns, target)
        updev = self.updev(returns, target)
        covar = self.covar(returns, benchmark)
        correl = self.correl(returns, benchmark)
        beta = self.beta(returns, benchmark)
        alpha = self.alpha(returns, benchmark)
        result = {"산술평균" : average,
              "CAGR": cagr,
              "표준편차": stdev,
              "하방 표준편차": downdev,
              "상방 표준편차": updev,
              "공분산": covar,
              "상관계수": correl,
                 "베타": beta,
                 "알파" : alpha}
        return result


class Tail():
    def __init__(self):
        self.annual=252
    # 왜도 : 수익률 분포의 비대칭 정도를 측정하기 위한 지표 
    def skewness(self, returns):
        return returns.skew()
    # 첨도 : 수익률 분포의 팻테일 정도를 측정하기 위한 지표
    def kurtosis(self, returns):
        return returns.kurtosis()
    # 공왜도 : 벤치마크 대비 전략의 수익률 분포가 얼마나 왜도를 가지고 있는지 알고 싶을 떄 사용하는 지표
    def coskewness(self, returns, benchmark):
        r_mean = returns.mean()
        b_mean = benchmark.mean()
        r_stdev = returns.std()
        b_stdev = benchmark.std()
        T = len(returns)        
        summation = ((returns - r_mean) * ((benchmark - b_mean) ** 2) / (r_stdev * (b_stdev ** 2))).sum()
        return (T / ((T - 1) * (T - 2))) * summation
    # 공첨도 : 공첨도는 어떤 두 수익률 데이터가 있을 때 벤치마크 대비 전략 수익률의 상대적인 첨도를 측정하기 위한 지표
    def cokurtosis(self, returns, benchmark):
        r_mean = returns.mean()
        b_mean = benchmark.mean()
        r_stdev = returns.std()
        b_stdev = benchmark.std()
        T = len(returns)
        summation = ((returns - r_mean) * ((benchmark - b_mean) ** 3) / (r_stdev * (b_stdev ** 3))).sum()
        return ((T * (T + 1)) / ((T - 1) * (T - 2) * (T - 3))) * summation - (3 * (T - 1) ** 2) / ((T - 2) * (T - 3))
    # 낙폭 : 현재 가격이 전 고점과 비교했을 때 얼마만큼의 손실을 보이고 있는지 나타내는 지표
    def drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        highwatermark = cumulative.cummax()
        drawdown = (cumulative / highwatermark) - 1
        return drawdown
    # 최대낙폭 : 백테스팅 기간 중 가장 큰 낙폭을 측정하기 위한 지표
    def maximum_drawdown(self, returns):
        return np.min(self.drawdown(returns))
    # 낙폭기간 : 과거 낙폭을 경험했던 여러 구간들에서 전 고점을 회복하기 전까지 낙폭이 유지된 기간
    def drawdown_duration(self, returns):
        drawdown = self.drawdown(returns)
        ddur = list(chain.from_iterable((np.arange(len(list(j))) + 1).tolist() if i==1 else [0] * len(list(j)) for i, j in groupby(drawdown != 0)))
        ddur = pd.DataFrame(ddur)
        ddur.index = returns.index
        return ddur
    # 최장낙폭기간 : 낙폭기간 중 가장 긴 기간을 알려주는 지표
    def maximum_drawdown_duration(self, returns):
        return self.drawdown_duration(returns).max()[0]
    # 역사적 VaR : 과거부터 수익률을 나열했을 때 하위 5% 지점에 있는 수익률
    def hVaR(self, returns, percentile=99):
        return returns.quantile(1 - percentile / 100)
    # 분석적 VaR : 정상적인 시장 상황 하에서 발생할 수 있는 최대 예상손실액
    def aVaR(self, returns, percentile=99):
        r_stdev = returns.std() 
        z_score = norm.ppf(percentile/100)
        return -z_score * r_stdev 
#     * np.sqrt(self.annual) # 연이율 기준
    # CVaR : VaR를 초과하는 손실률들의 평균값
    def CVaR(self, returns, percentile=99):
        return returns[returns < self.hVaR(returns, percentile)].mean()
    # 한번에 출력
    def print_result(self, returns, benchmark, percentile=99):
        skew = self.skewness(returns)
        kurt = self.kurtosis(returns)
        coskew = self.coskewness(returns, benchmark)
        cokurt = self.cokurtosis(returns, benchmark)
        mdd = self.maximum_drawdown(returns)
        mddur = self.maximum_drawdown_duration(returns)
        hvar = self.hVaR(returns, percentile)
        avar = self.aVaR(returns)
        cvar = self.CVaR(returns, percentile)
        
        result = {"Skewness" : skew,
                  "Kurtosis" : kurt,
                  "Co-Skewness" : coskew,
                  "Co-Kurtosis" : cokurt,
                  "Maximum Drawdown" : mdd,
                  "Maximum Drawdown Duration" : mddur,
                  "99% HVaR" : hvar,
                  "99% AVaR" : avar,
                  "99% CVaR" : cvar}
        return result


class Performance(Core, Tail):    # Inherit from Core(), Tail()  
    #추적 오차
    def track_error(self, returns, benchmark):
        return self.stdev(returns - benchmark)
    # 정보 비율 : 
    def information_ratio(self, returns, benchmark):
        return self.average(returns - benchmark) / self.stdev(returns - benchmark)
    # 샤프 비율 : 전략의 성과를 평가하기 위한 지표로 초과 수익률의 평균을 변동성으로 나눔
    def sharpe_ratio(self, returns):
        return self.average(returns - r_f) / self.stdev(returns)
    # 조정 샤프 비율 : 초과 수익률의 왜도와 첨도를 반영하여 테일 리스크가 반영된 샤프비율
    def adjusted_sharpe_ratio(self, returns):
        skewness = self.skewness(returns)
        kurtosis = self.kurtosis(returns)
        sharpe_ratio = self.sharpe_ratio(returns)
        return sharpe_ratio * (1 + skewness * sharpe_ratio / 6 - kurtosis * (sharpe_ratio ** 2) / 24)
    # 소르티노 비율 : 분모에 표준편차가 아닌 하방 표준편차를 적용한 성과 지표
    def sortino_ratio(self, returns, benchmark, target=0.0):
        return self.average(returns - benchmark) / self.downdev(returns, target)
    # 칼마 비율 : 최대낙폭 대비 전략의 초과수익률을 측정하는 지표
    def calmar_ratio(self, returns, benchmark):
        return -self.average(returns - benchmark) / self.maximum_drawdown(returns)
    # 트레이너 비율 : 위험 한 단위를 받고 얻은 초과성과가 얼마인지를 측정하는 성과지표
    def treynor_ratio(self, returns, benchmark):
        return self.average(returns - benchmark) / self.beta(returns, benchmark)
    # VaR 대비 성과 비율 : 위험 지표 중 하나인 VaR 대비 수익의 비율을 나타내는 지표
    def reward_to_VaR_ratio(self, returns, benchmark):
        return -self.average(returns - benchmark) / self.hVaR(returns)
    # CVaR 대비 성과 비율 : VaR 대신 분모에 CVaR를 사용한 지표
    def reward_to_CVaR_ratio(self, returns, benchmark):
        return -self.average(returns - benchmark) / self.CVaR(returns)
    # 승률 : 전체 매매 횟수 중 수익을 얻은 거래의 비율
    def hit_ratio(self, returns):
        return len(returns[returns > 0]) / (len(returns[returns > 0]) + len(returns[returns < 0]))
    # 손익비 : 전략을 운용할 때 1회당 평균 손실 금액 대비 1회당 평균 이익 금액의 비율
    def gain_to_pain_ratio(self, returns):
        return - returns[returns > 0].sum() / returns[returns < 0].sum()
    # 한번에 출력
    def print_result(self, returns, benchmark, target=0.0):
        track_error = self.track_error(returns, benchmark)
        information_ratio = self.information_ratio(returns, benchmark)
        sr = self.sharpe_ratio(returns)
        asr = self.adjusted_sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns, benchmark, target)
        calmar = self.calmar_ratio(returns, benchmark)
        treynor = self.treynor_ratio(returns, benchmark)
        varratio = self.reward_to_VaR_ratio(returns, benchmark)
        cvarratio = self.reward_to_CVaR_ratio(returns, benchmark)
        hitratio = self.hit_ratio(returns)
        gpratio = self.gain_to_pain_ratio(returns)
        
        result = {"Track Error" : track_error,
                  "Information Ratio" : information_ratio,
                  "Sharpe Ratio" : sr,
                  "Adjusted Sharpe Ratio" : asr,
                  "Sortino Ratio" : sortino,
                  "Calmar Ratio" : calmar,
                  "Treynor Ratio" : treynor,
                  "Reward-to-VaR Ratio" : varratio,
                  "Reward-to-CVaR Ratio" : cvarratio,
                  "Hit Ratio" : hitratio,
                  "Gain-to-Pain Ratio" : gpratio}
        return result


# 무위험 금리(CD91) 불러오기
def get_product(KEY, STAT_CD, PERIOD, START_DATE, END_DATE):
    # 파이썬에서 인터넷을 연결하기 위해 urllib 패키지 사용. urlopen함수는 지정한 url과 소켓 통신을 할 수 있도록 자동 연결해줌
    # 인증키, 추출할 통계지표의 코드, 기간 단위, 데이터 시작일, 종료일, 통계항목 코드
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{KEY}/xml/kr/1/30000/{STAT_CD}/{PERIOD}/{START_DATE}/{END_DATE}/{ITEM_CODE}"
    response = requests.get(url).content.decode('utf-8')
    xml_obj = BeautifulSoup(response, 'lxml-xml')
    rows = xml_obj.findAll("row")
    return rows


if __name__ == "__main__":
    data_dict = {'817Y002' : '시장금리(일별)'}  # 파라미터 정의 :: 추출하고자 하는 통계지표 disc type - {통계지표 코드: 통계지표명}
    KEY = 'DSFS58V3CLRQ4KOKWNVH'    # 인증키
    ITEM_CODE = '010502000'         # 통계항목 코드
    PERIOD = 'D'
    START_DATE = '20120101'
    END_DATE = '20230119'

    # API의 반환(출력)값 중 저장하고자 하는 항목(item) 리스트
    item_list = [
    'STAT_CODE' # 통계표코드
    , 'STAT_NAME' # 통계명
    , 'ITEM_CODE1' # 통계항목1코드
    , 'ITEM_NAME1' # 통계항목명1
    , 'ITEM_CODE2' # 통계항목2코드
    , 'ITEM_NAME2' # 통계항목명2
    , 'ITEM_CODE3' # 통계항목3코드
    , 'ITEM_NAME3' # 통계항목명3
    , 'UNIT_NAME' # 단위
    , 'TIME' # 시점
    , 'DATA_VALUE'# 값
    ]

    # 결과치를 담을 빈 리스트 생성
    result_list = list()

    # API를 순차적으로 호출하고 값을 담는 for loop 구문
    for k in data_dict.keys():
        rows = get_product(KEY, k, PERIOD, START_DATE, END_DATE)
        print(len(rows)) # 수집해야 할 데이터의 row가 총 몇 개인지 출력

        for p in range(0, len(rows)):
            info_list = list()
            
            for i in item_list:
                try:
                    individual_info = rows[p].find(i).text # 만약 반환 중 error가 발생하면
                except:
                    individual_info = "" # 해당 항목은 공란으로 채운다
            
                info_list.append(individual_info)
            result_list.append(info_list)
        result_list


    # 결과 리스트를 DataFrame으로 변환 + 컬럼명 지정
    result_df = pd.DataFrame(result_list, columns=[
    '통계표코드'
    , '통계명'
    , '통계항목1코드'
    , '통계항목명1'
    , '통계항목2코드'
    , '통계항목명2'
    , '통계항목3코드'
    , '통계항목명3'
    , '단위'
    , '시점'
    , '값'
    ]).drop_duplicates() # 중복된 row 제거

    numeric_df = pd.to_numeric(result_df['값'])
    r_f = numeric_df.mean()
    r_f = (r_f / 100) / 255