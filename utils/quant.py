import os
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from utils.metrics import get_metrics

def check_IFRS(x):
    if x == 'N/A(IFRS)':
        return np.NaN
    else:
        return x
    
def get_finance_data(path):
    #  [코드 4.6] 재무 데이터 전처리하는 함수 (CH4. 전략 구현하기.ipynb)
    data_path = path
    raw_data = pd.read_excel(data_path)
    raw_data=raw_data.set_index('Unnamed: 0')
    big_col = list(raw_data.columns)
    small_col = list(raw_data.iloc[0])
    
    new_big_col = []
    for num, col in tqdm(enumerate(big_col)):
        if 'Unnamed' in col:
            new_big_col.append(new_big_col[num-1])
        else:
            new_big_col.append(big_col[num])
            
    raw_data.columns = [new_big_col, small_col]
    clean_df = raw_data.loc[ raw_data.index.dropna() ]
    return clean_df

def select_code_by_price(price_df, data_df, start_date):
    #사용된 투자지표에 있는 종목들이 가격데이터에 없으면 가격데이터에서 종목 삭제
    new_code_list = []
    for code in price_df[start_date].iloc[0].dropna().index:
        if 'A'+code in list(data_df.index):
            new_code_list.append('A'+code)
        
        #new_code_list.append('A' + code)
        
    selected_df =  data_df.loc[new_code_list]
    return selected_df

def get_strategy_date(start_date):
    # [코드 5.24] 백테스트 시작날짜가 주어지면 전략 기준 날짜를 계산하는 함수 (Ch5. 백테스트.ipynb)
    temp_year = int(start_date.split('-')[0])
    temp_month = start_date.split('-')[1]
    if temp_month in '1 2 3 4 5'.split(' '):
        strategy_date = str(temp_year - 2) + '/12'
    else:
        strategy_date = str(temp_year - 1) + '/12'
    return strategy_date
    
    
def low_per(invest_df, index_date, num):
    invest_df[(index_date, 'PER')] = pd.to_numeric(invest_df[(index_date, 'PER')])
    per_sorted = invest_df.sort_values(by=(index_date, 'PER'))
    return per_sorted[index_date][:num]


def high_roa(fr_df, index_date, num):
    fr_df[(index_date, 'ROA')] = fr_df[(index_date, 'ROA')].apply(check_IFRS)
    fr_df[(index_date, 'ROA')] = pd.to_numeric(fr_df[(index_date, 'ROA')] )
    sorted_roa = fr_df.sort_values(by=(index_date, 'ROA'), ascending=False)
    return sorted_roa[index_date][:num]


def magic_formula(fr_df, invest_df, index_date, num):
    #  [코드 4.22] 마법공식 함수로 만들기 (CH4. 전략 구현하기.ipynb)
    per = low_per(invest_df, index_date, None)
    roa = high_roa(fr_df, index_date, None)
    per['per순위'] = per['PER'].rank()
    roa['roa순위'] = roa['ROA'].rank(ascending=False)
    magic = pd.merge(per, roa, how='outer', left_index=True, right_index=True)
    magic['마법공식 순위'] = (magic['per순위'] + magic['roa순위']).rank().sort_values()
    magic = magic.sort_values(by='마법공식 순위')
    return magic[:num]


def get_value_rank(invest_df, value_type, index_date, num):
    #  [코드 4.23] 저평가 지수를 기준으로 정렬하여 순위 만들어 주는 함수 (CH4. 전략 구현하기.ipynb)
    invest_df[(index_date,  value_type)] = pd.to_numeric(invest_df[(index_date,  value_type)])
    value_sorted = invest_df.sort_values(by=(index_date,  value_type))[index_date]
    value_sorted[  value_type + '순위'] = value_sorted[value_type].rank()
    return value_sorted[[value_type, value_type + '순위']][:num]


def get_fscore(fs_df, index_date, num):
    #  [코드 4.29] F-score 함수(CH4. 전략 구현하기.ipynb)
    fscore_df = fs_df[index_date]
    fscore_df['당기순이익점수'] = fscore_df['당기순이익'] > 0
    fscore_df['영업활동점수'] = fscore_df['영업활동으로인한현금흐름'] > 0
    fscore_df['더큰영업활동점수'] = fscore_df['영업활동으로인한현금흐름'] > fscore_df['당기순이익']
    fscore_df['종합점수'] = fscore_df[['당기순이익점수', '영업활동점수', '더큰영업활동점수']].sum(axis=1)
    fscore_df = fscore_df[fscore_df['종합점수'] == 3]
    return fscore_df[:num]


def make_value_combo(value_list, invest_df, index_date, num):
    #  [코드 4.25] 저평가 지표 조합 함수 (CH4. 전략 구현하기.ipynb)
    for i, value in enumerate(value_list):
        temp_df = get_value_rank(invest_df, value, index_date, None)
        if i == 0:
            value_combo_df = temp_df
            rank_combo = temp_df[value + '순위']
        else:
            value_combo_df = pd.merge(value_combo_df, temp_df, how='outer', left_index=True, right_index=True)
            rank_combo = rank_combo + temp_df[value + '순위']
    
    value_combo_df['종합순위'] = rank_combo.rank()
    value_combo_df = value_combo_df.sort_values(by='종합순위')
    return value_combo_df[:num]


def get_value_quality(invest_df, fs_df, index_date, num):
    #  [코드 4.39] 저평가 + Fscore 함수화 (CH4. 전략 구현하기.ipynb)
    value = make_value_combo(['PER', 'PBR', 'PSR', 'PCR'], invest_df, index_date, None)
    quality = get_fscore(fs_df, index_date, None)
    value_quality = pd.merge(value, quality, how='outer', left_index=True, right_index=True)
    value_quality_filtered = value_quality[value_quality['종합점수'] == 3]
    vq_df = value_quality_filtered.sort_values(by='종합순위')
    return vq_df[:num]


def high_roa(fr_df, index_date, num):
    fr_df[(index_date, 'ROA')] = fr_df[(index_date, 'ROA')].apply(check_IFRS)
    fr_df[(index_date, 'ROA')] = pd.to_numeric(fr_df[(index_date, 'ROA')] )
    sorted_roa = fr_df.sort_values(by=(index_date, 'ROA'), ascending=False)
    return sorted_roa[index_date][:num]


def backtest_re(strategy, start_date, end_date, initial_money, price_df, invest_df, fr_df, fs_df, num, value_type=None, value_list=None, date_range=None):
    # [코드 5.32] 리밸런싱 백테스트 함수화 (Ch5. 백테스트.ipynb)
    
    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])

    total_df = 0
    for temp in range(start_year, end_year):
        this_term_start = str(temp) + '-' + start_date.split('-')[1]
        this_term_end = str(temp+1) + '-' + start_date.split('-')[1]
        strategy_date = get_strategy_date(this_term_start)
        
        if strategy.__name__ == 'high_roa':
            #low_per = get_value_rank(invest_df, 'PER', strategy_date, 20)
            # st_df = strategy(invest_df, value_type, strategy_date, num)

            st_df = strategy(select_code_by_price(price_df, fr_df, this_term_start), strategy_date, num)
        elif strategy.__name__ == 'magic_formula':
            #(fr_df, invest_df, index_date, num)
            #st_df = strategy(fr_df, invest_df, strategy_date, num)
            st_df = strategy(fr_df, select_code_by_price(price_df, invest_df, this_term_start), strategy_date, num)
        elif strategy.__name__ == 'get_value_rank':
            #get_value_rank(invest_df, value_type, index_date, num)
            #st_df = strategy(invest_df, value_type, strategy_date, num)
            st_df = strategy(select_code_by_price(price_df, invest_df, this_term_start), value_type, strategy_date, num)
        elif strategy.__name__ == 'make_value_combo':
            #st_df = strategy(value_list, invest_df, strategy_date, num)

            st_df = strategy(value_list, select_code_by_price(price_df, invest_df, this_term_start), strategy_date, num)
        elif strategy.__name__ == 'get_fscore':
            #st_df = strategy(fs_df, strategy_date, num)

            st_df = strategy(select_code_by_price(price_df, fs_df, this_term_start), strategy_date, num)
        elif strategy.__name__ == 'get_momentum_rank':
            #st_df = strategy(price_df, price_df[this_term_start].index[0] , date_range, num)

            st_df = strategy(price_df, price_df[this_term_start].index[0] , date_range, num)
        elif strategy.__name__ == 'get_value_quality':
            #st_df = strategy(invest_df, fs_df, strategy_date, num)
            st_df = strategy(
                select_code_by_price(price_df, invest_df, this_term_start), 
                select_code_by_price(price_df, fs_df, this_term_start),
                strategy_date, num
            )
        
        backtest = backtest_beta(price_df, st_df, this_term_start, this_term_end, initial_money)
        temp_end = backtest[this_term_end].index[0]
        backtest = backtest[:temp_end]
        initial_money =  backtest['종합포트폴리오'][-1]
        if temp == start_year:
            total_df = backtest
        else:
            total_df = pd.concat([total_df[:-1], backtest])

    total_df ['일변화율'] = total_df ['종합포트폴리오'].pct_change()
    total_df ['총변화율'] = total_df ['종합포트폴리오']/ total_df ['종합포트폴리오'][0] - 1
    
    return total_df


def backtest_beta(price_df, strategy_df, start_date, end_date, initial_money):
    #  [코드 5.12] 백테스트 함수 버젼1 (Ch5. 백테스트.ipynb)
    code_list = []
    for code in strategy_df.index:
        code_list.append(code.replace('A',''))

    strategy_price = price_df[code_list][start_date:end_date]

    pf_stock_num = {}
    stock_amount = 0
    stock_pf = 0
    each_money = initial_money / len(strategy_df)
    for code in strategy_price.columns:
        temp = int( each_money / strategy_price[code][0] )
        pf_stock_num[code] = temp
        stock_amount = stock_amount + temp * strategy_price[code][0]
        stock_pf = stock_pf + strategy_price[code] * pf_stock_num[code]

    cash_amount = initial_money - stock_amount

    backtest_df = pd.DataFrame({'주식포트폴리오':stock_pf})
    backtest_df['현금포트폴리오'] = [cash_amount] * len(backtest_df)
    backtest_df['종합포트폴리오'] = backtest_df['주식포트폴리오'] + backtest_df['현금포트폴리오']
    backtest_df['일변화율'] = backtest_df['종합포트폴리오'].pct_change()
    backtest_df['총변화율'] = backtest_df['종합포트폴리오']/initial_money - 1
    
    return backtest_df


def get_mdd(back_test_df):
    # [코드 5.40] MDD 함수화 (Ch5. 백테스트.ipynb)
    max_list = [0]
    mdd_list = [0]

    for i in back_test_df.index[1:]:
        max_list.append(back_test_df['총변화율'][:i].max())
        if max_list[-1] > max_list[-2]:
            mdd_list.append(0)
        else:
            mdd_list.append(min(back_test_df['총변화율'][i] - max_list[-1], mdd_list[-1])   )

    back_test_df['max'] = max_list
    back_test_df['MDD'] = mdd_list
    
    return back_test_df


def get_month_range(year, month):
    date = datetime(year=year, month=month, day=1).date()
    monthrange = calendar.monthrange(date.year, date.month)
    return monthrange[1]


def quanting(info, fs_df, fr_df, invest_df, price_df):
    start_date = f"{int(info['startYear'])}-{int(info['startMonth'])}"
    end_date = f"{int(info['endYear'])}-{int(info['endMonth'])}"
    initial_money = int(info['moneyToStart'])*10000
    strategy_name = info['strategy']
    strategies = {
        'get_value_rank': get_value_rank,
        'make_value_combo': make_value_combo,
        'magic_formula': magic_formula,
        'high_roa': high_roa,
        'get_value_quality': get_value_quality
    }
    strategy = strategies[strategy_name]
    if strategy_name == 'get_value_rank':
        result_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, invest_df, fr_df, fs_df, 20, value_type='PER')    # 하나 선택
    elif strategy_name == 'make_value_combo':
        result_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, invest_df, fr_df, fs_df, 20, value_list=['PCR', 'PBR', 'PER', 'PSR'])    # 복수 선택
    elif strategy_name == 'magic_formula':
        result_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, invest_df, fr_df, fs_df, 20)
    elif strategy_name == 'high_roa':
        result_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, invest_df, fr_df, fs_df, 20)
    elif strategy_name == 'get_value_quality':
        result_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, invest_df, fr_df, fs_df, 20)
        
    result_df.fillna(0)
    print(result_df)
    # 매월 기간 수익률 구하기
    m_l_df = result_df['종합포트폴리오'].resample('M').last()
    m_f_df = result_df['종합포트폴리오'].resample('M').last().shift(1)
    m_df = m_l_df.div(m_f_df)-1
    
    # 매년 기간 수익률 구하기
    y_l_df = result_df['종합포트폴리오'].resample('Y').last()
    y_f_df = result_df['종합포트폴리오'].resample('Y').last().shift(1)
    y_df = y_l_df.div(y_f_df)-1
    
    # 포트폴리오 지표 계산
    s = start_date.split('-')
    start_day = f"{s[0]}-{s[1].zfill(2)}-01"
    e = end_date.split('-')
    end_day = f"{e[0]}-{e[1].zfill(2)}-{get_month_range(int(e[0]), int(e[1]))}"
    metrics = get_metrics(start_day, end_day, result_df['일변화율'].squeeze())
    
    print(result_df)
    print(m_df)
    print(y_df)
    print(metrics)
    return result_df, m_df, y_df, metrics

if __name__ == "__main__":
    # [코드 5.33] 저PER과 저PBR 비교 (Ch5. 백테스트.ipynb)
    start_date = '2016-6'
    end_date = '2022-6'
    initial_money = 100000000
    strategy = get_value_quality 
    # 함수 명 [
        # get_value_rank(단일 PBR, PER, PSR, PCR 지정 가능),
        # make_value_combo(복수 조합 지정 가능) => get_value_quality(f-score + make value combo), 
        # magic_formula(저 PER + 고 ROA)
        # high_roa(고 ROA)
        # ]
    
    fs_path = 'utils/data/재무제표데이터.xlsx'
    fs_df = get_finance_data(fs_path)
    fr_path = 'utils/data/재무비율데이터.xlsx'
    fr_df = get_finance_data(fr_path)
    invest_path = 'utils/data/투자지표데이터.xlsx'
    invest_df = get_finance_data(invest_path)
    price_path = 'utils/data/가격데이터.csv'
    price_df = pd.read_csv(price_path)
    price_df = price_df.set_index('Unnamed: 0')
    price_df.index = [price_df.index[i][:10] for i in range(len(price_df['060310']))]
    price_df.index = pd.to_datetime(price_df.index)

    
    # get_value_rank_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, fr_df, fs_df, 20, value_type='PER')    # 하나 선택
    # make_value_combo_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, fr_df, fs_df, 20, value_list=['PCR', 'PBR', 'PER', 'PSR'])    # 복수 선택
    # magic_formula_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, fr_df, fs_df, 20)
    # high_roa_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, fr_df, fs_df, 20)
    get_value_quality_df = backtest_re(strategy, start_date, end_date, initial_money, price_df, fr_df, fs_df, 20)
    back_test_result1 = get_value_quality_df
    print(back_test_result1)
    plt.figure(figsize=(10, 6))
    back_test_result1['총변화율'].plot(label='PER')
    # back_test_result2['총변화율'].plot(label='PBR')
    plt.legend()
    plt.show()
