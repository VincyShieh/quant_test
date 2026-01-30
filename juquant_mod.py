# 克隆自聚宽文章：https://www.joinquant.com/post/48819
# 标题：全天候轮动
# 作者：MarioC

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import pickle
import talib
import warnings
import random

np.random.seed(42)
random.seed(42)

warnings.filterwarnings("ignore")
# 初始化函数
def initialize(context):
    # 设定基准
    set_benchmark('000300.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                             close_today_commission=0, min_commission=5), type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    # 初始化全局变量
    g.no_trading_today_signal = False
    g.stock_num = 3  # 原版参数
    g.hold_list = []  # 当前持仓的全部股票
    g.yesterday_HL_list = []  # 记录持仓中昨日涨停的股票
    g.foreign_ETF = [
        '518880.XSHG',
        '513030.XSHG',
        '513100.XSHG',
        '164824.XSHE',
        '159866.XSHE',
        ]
    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05')
    run_monthly(monthly_adjustment, 2, '9:30')  # 改为2号避免节假日
    run_daily(stop_loss, '14:00')

def prepare_stock_list(context):
    # 获取已持有列表
    g.hold_list = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)
    # 获取昨日涨停列表
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close', 'high_limit'],
                       count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []
        
def stop_loss(context):
    """
    原版简单止损逻辑：
    1. 涨停保护（原有逻辑）
    2. 硬止损：亏损超过8%
    3. 补仓逻辑（原有）
    """
    num = 0
    now_time = context.current_dt

    # 1. 涨停保护（原有逻辑）
    if g.yesterday_HL_list != []:
        for stock in g.yesterday_HL_list:
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close', 'high_limit'],
                                     skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                log.info("[%s]涨停打开，卖出" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
                num = num+1
            else:
                log.info("[%s]涨停，继续持有" % (stock))

    # 2. 简单止损逻辑（原版）
    SS=[]
    S=[]
    for stock in g.hold_list:
        if stock in list(context.portfolio.positions.keys()):
            if context.portfolio.positions[stock].price < context.portfolio.positions[stock].avg_cost * 0.92:
                order_target_value(stock, 0)
                log.debug("止损 Selling out %s" % (stock))
                num = num+1
            else:
                S.append(stock)
                NOW = (context.portfolio.positions[stock].price - context.portfolio.positions[stock].avg_cost)/context.portfolio.positions[stock].avg_cost
                SS.append(np.array(NOW))

    # 3. 补仓逻辑（原有）
    if num >= 1:
        if len(SS) > 0:
            num = 3
            min_values = sorted(SS)[:num]
            min_indices = [SS.index(value) for value in min_values]
            min_strings = [S[index] for index in min_indices]
            cash = context.portfolio.cash/num
            for ss in min_strings:
                order_value(ss, cash)
                log.debug("补跌最多的N支 Order %s" % (ss))


def filter_industry_diversity(context, stock_list, max_per_industry=1):
    """
    行业分散度约束：每个行业最多max_per_industry只股票
    避免单一行业风险暴露过大
    使用 get_fundamentals 查询行业信息，更加稳健
    """
    if len(stock_list) == 0:
        return []

    # 使用 get_fundamentals 查询行业信息（更稳健的方式）
    try:
        industry_df = get_fundamentals(
            query(
                valuation.code,
                industry.sw_l1,  # 申万一级行业
            ).filter(
                valuation.code.in_(stock_list)
            ),
            date=context.previous_date
        )

        # 检查查询结果是否为空
        if industry_df is None or len(industry_df) == 0:
            log.warning("行业数据查询为空，跳过行业分散检查")
            return stock_list

        # 建立股票到行业的映射
        stock_industry_map = {}
        for _, row in industry_df.iterrows():
            stock_industry_map[row['code']] = row['sw_l1']

        industries = {}
        filtered = []

        for stock in stock_list:
            # 获取该股票的行业
            industry = stock_industry_map.get(stock)

            if industry is None:
                # 如果无法获取行业信息，仍然保留该股票
                filtered.append(stock)
                continue

            # 检查该行业是否已达上限
            if industries.get(industry, 0) < max_per_industry:
                industries[industry] = industries.get(industry, 0) + 1
                filtered.append(stock)
            else:
                log.debug(f"{stock} 跳过，行业{industry}已达上限({industries[industry]}/{max_per_industry})")

        log.info(f"行业分散过滤: 从{len(stock_list)}只筛选到{len(filtered)}只，行业分布: {industries}")
        return filtered

    except Exception as e:
        # 如果查询失败，返回原始列表（避免因错误导致策略中断）
        import traceback
        log.error(f"行业分散过滤失败: {e}，跳过行业分散检查")
        log.error(f"详细错误: {traceback.format_exc()}")
        return stock_list


def check_market_environment(context):
    """
    市场环境判断：判断是否处于熊市
    使用沪深300指数的60日均线和跌幅来判断
    """
    try:
        # 获取沪深300过去60天数据
        index_data = get_price('000300.XSHG', end_date=context.previous_date,
                            frequency='daily', fields=['close'],
                            count=61, panel=False, fill_paused=False)
        
        if len(index_data) < 61:
            # 数据不足，默认不是熊市
            return False
        
        # 计算60日平均价格
        ma60 = index_data['close'].iloc[:-1].mean()
        current_price = index_data['close'].iloc[-1]
        price_60d_ago = index_data['close'].iloc[-61]
        
        # 计算跌幅
        decline_60d = (current_price - price_60d_ago) / price_60d_ago
        
        # 判断是否在熊市：
        # 1. 当前价格低于60日均线
        # 2. 60天跌幅超过10%
        is_bear_market = (current_price < ma60) and (decline_60d < -0.1)
        
        if is_bear_market:
            log.info(f"市场环境不佳: 当前价格{current_price:.2f} < 60日均线{ma60:.2f}, 跌幅{decline_60d*100:.2f}%")
        
        return is_bear_market
        
    except Exception as e:
        log.error(f"判断市场环境失败: {e}")
        return False


def add_northbound_filter(context, stock_list):
    """
    北向资金因子：过滤有北向资金增持的股票
    外资偏好股票通常表现更好
    """
    if len(stock_list) == 0:
        return []

    try:
        # 查询北向资金持股数据
        # 注意：这里使用聚宽的北向资金相关指标
        nb_df = get_fundamentals(
            query(
                valuation.code,
                valuation.circulating_market_cap,  # 流通市值
            ).filter(
                valuation.code.in_(stock_list)
            ),
            date=context.previous_date
        )

        if nb_df is None or len(nb_df) == 0:
            log.warning("北向资金数据查询为空，跳过北向资金过滤")
            return stock_list

        # 简化逻辑：选择流通市值较大的前50%股票
        # (实际北向资金数据可能需要付费接口，这里用市值替代)
        nb_df = nb_df.sort_values('circulating_market_cap', ascending=False)
        top_half_count = len(nb_df) // 2
        filtered_stocks = nb_df['code'].head(max(top_half_count, 1)).tolist()

        log.info(f"北向资金过滤（按市值替代）: 从{len(stock_list)}只筛选到{len(filtered_stocks)}只")
        return filtered_stocks

    except Exception as e:
        import traceback
        log.error(f"北向资金过滤失败: {e}，跳过北向资金检查")
        log.debug(f"详细错误: {traceback.format_exc()}")
        return stock_list


def SMALL(context,choice):
    # 第一层：基本面筛选
    df = get_fundamentals(query(
        valuation.code,
        indicator.roe,
        indicator.roa,
        valuation.pe_ratio,
        valuation.pb_ratio,
        indicator.inc_revenue_year_on_year,
        indicator.inc_net_profit_year_on_year,
        valuation.turnover_ratio,
    ).filter(
        valuation.code.in_(choice),
        indicator.roe > 0.12,  # 降低ROE要求，从15%到12%
        indicator.roa > 0.08,  # 降低ROA要求，从10%到8%
        valuation.pe_ratio.between(5, 50),  # 控制估值，避免过贵
        valuation.pb_ratio.between(1, 8),  # 控制市净率
        indicator.inc_revenue_year_on_year > 0.1,  # 收入增长>10%
        indicator.inc_net_profit_year_on_year > 0.1,  # 利润增长>10%
        valuation.turnover_ratio > 3,  # 流动性过滤，换手率>3%
    )).set_index('code').index.tolist()

    if len(df) == 0:
        return []

    # 第二层：技术面筛选（20日动量）
    momentum_df = get_price(df, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=21, panel=False, fill_paused=False)
    momentum_df = momentum_df.pivot(index='time', columns='code', values='close')
    # 防止除零错误
    momentum_df_first = momentum_df.iloc[0]
    momentum_df_first[momentum_df_first == 0] = np.nan
    momentum = (momentum_df.iloc[-1] / momentum_df_first - 1) * 100

    # 过滤：动量>0 且 排名前50%
    momentum_sorted = momentum.sort_values(ascending=False)
    good_momentum_stocks = momentum_sorted[momentum_sorted > 0].index.tolist()[:len(momentum_sorted)//2]

    # 交叉筛选
    df = [s for s in df if s in good_momentum_stocks]

    if len(df) == 0:
        return []

    # 第三层：按市值升序排序，取前g.stock_num*3只
    q = query(
        valuation.code
    ).filter(
        valuation.code.in_(df)
    ).order_by(
        valuation.market_cap.asc()
    )
    final_list = list(get_fundamentals(q).code)[:g.stock_num*3]

    return final_list
    
def BIG(context,choice):
    # 第一层：基本面筛选，增加分红和负债率约束
    df = get_fundamentals(query(
        valuation.code,
        valuation.pe_ratio_lyr,
        valuation.ps_ratio,
        valuation.pcf_ratio,
        indicator.eps,
        indicator.roe,
        indicator.net_profit_margin,
        indicator.gross_profit_margin,
        indicator.inc_revenue_year_on_year,
        balance.total_liability,
        balance.total_sheet_owner_equities,
    ).filter(
        valuation.code.in_(choice),
        valuation.pe_ratio_lyr.between(5, 30),  # 提高PE下限，避免过便宜的公司
        valuation.ps_ratio.between(1, 8),
        valuation.pcf_ratio < 10,
        indicator.eps > 0.3,
        indicator.roe > 0.1,
        indicator.net_profit_margin > 0.1,
        indicator.gross_profit_margin > 0.3,
        indicator.inc_revenue_year_on_year > 0.15,  # 降低增长要求，从25%到15%
        (balance.total_liability / balance.total_sheet_owner_equities) < 1.5,  # 负债率控制
    )).set_index('code').index.tolist()

    if len(df) == 0:
        return []

    # 第二层：技术面筛选（60日动量）
    momentum_df = get_price(df, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=61, panel=False, fill_paused=False)
    momentum_df = momentum_df.pivot(index='time', columns='code', values='close')
    # 防止除零错误
    momentum_df_first = momentum_df.iloc[0]
    momentum_df_first[momentum_df_first == 0] = np.nan
    momentum = (momentum_df.iloc[-1] / momentum_df_first - 1) * 100

    # 过滤：60日动量>5%
    good_momentum_stocks = momentum[momentum > 5].index.tolist()

    # 交叉筛选
    df = [s for s in df if s in good_momentum_stocks]

    if len(df) == 0:
        return []

    # 第三层：综合评分排序（ROE*60% + 动量*40%）
    df_fundamentals = get_fundamentals(query(
        valuation.code,
        indicator.roe,
    ).filter(valuation.code.in_(df)))

    # 归一化评分
    max_roe = df_fundamentals['roe'].max()

    df_fundamentals['score'] = (
        df_fundamentals['roe'] / max_roe * 0.6 +
        (momentum[df_fundamentals['code']].values / momentum.max() * 0.4)
    )

    df_fundamentals = df_fundamentals.sort_values('score', ascending=False)
    final_list = df_fundamentals['code'].tolist()[:g.stock_num]

    return final_list
def ROIC_BIG(context,choice):
    # 第一层：基本面筛选，放宽部分条件
    df = get_fundamentals(query(
            valuation.code,
        ).filter(
            valuation.code.in_(choice),
            valuation.market_cap > 200,  # 降低市值要求，从300亿到200亿
            valuation.pe_ratio.between(0, 50),
            indicator.eps > 0.1,  # 降低EPS要求，从0.12到0.1
            indicator.roa > 0.12,  # 降低ROA要求，从0.15到0.12
            (balance.total_liability / balance.total_sheet_owner_equities) < 0.7,  # 放宽负债率，从0.5到0.7
            indicator.inc_total_revenue_year_on_year > 0.2,  # 降低增长要求，从0.3到0.2
            indicator.inc_revenue_year_on_year > 0.15,  # 降低增长要求，从0.2到0.15
            balance.retained_profit > 0,
        )).set_index('code').index.tolist()

    if len(df) == 0:
        return []

    # 第二层：ROIC筛选
    yesterday = context.previous_date
    roic_stocks = []
    for stock in df:
        roic_df = get_factor_values(stock, 'roic_ttm', end_date=yesterday, count=1)
        roic = roic_df['roic_ttm'].iloc[0, 0]
        if roic > 0.08:
            roic_stocks.append(stock)

    df = roic_stocks

    if len(df) == 0:
        return []

    # 第三层：综合评分排序（ROIC*50% + ROE*30% + 动量*20%）
    df_fundamentals = get_fundamentals(query(
        valuation.code,
        indicator.roe,
    ).filter(valuation.code.in_(df)))

    # 获取ROIC
    roic_dict = {}
    for stock in df:
        roic = get_factor_values(stock, 'roic_ttm', end_date=context.previous_date, count=1)['roic_ttm'].iloc[0, 0]
        roic_dict[stock] = roic

    # 获取60日动量
    momentum_df = get_price(df, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=61, panel=False, fill_paused=False)
    momentum_df = momentum_df.pivot(index='time', columns='code', values='close')
    # 防止除零错误
    momentum_df_first = momentum_df.iloc[0]
    momentum_df_first[momentum_df_first == 0] = np.nan
    momentum = (momentum_df.iloc[-1] / momentum_df_first - 1) * 100

    # 合并数据
    score_df = pd.DataFrame({'code': df})
    score_df['roic'] = score_df['code'].map(roic_dict)
    score_df['roe'] = score_df['code'].map(df_fundamentals.set_index('code')['roe'])
    score_df['momentum'] = score_df['code'].map(momentum)

    # 归一化评分
    max_roic = score_df['roic'].max()
    max_roe = score_df['roe'].max()
    max_momentum = score_df['momentum'].max() - score_df['momentum'].min()

    if max_roic > 0:
        score_df['roic_score'] = score_df['roic'] / max_roic
    else:
        score_df['roic_score'] = 0

    if max_roe > 0:
        score_df['roe_score'] = score_df['roe'] / max_roe
    else:
        score_df['roe_score'] = 0

    if max_momentum > 0:
        score_df['momentum_score'] = (score_df['momentum'] - score_df['momentum'].min()) / max_momentum
    else:
        score_df['momentum_score'] = 0

    # 综合评分
    score_df['final_score'] = (
        score_df['roic_score'] * 0.5 +
        score_df['roe_score'] * 0.3 +
        score_df['momentum_score'] * 0.2
    )

    score_df = score_df.sort_values('final_score', ascending=False)
    final_list = score_df['code'].tolist()[:g.stock_num]

    return final_list
def BM(context,choice):
    # 第一层：基本面筛选，优化筛选逻辑
    BM_list = get_fundamentals(query(
            valuation.code,
            valuation.pe_ratio,  # 增加PE
            valuation.ps_ratio,  # 增加PS
            valuation.pcf_ratio,
            indicator.eps,
            indicator.roe,
            indicator.net_profit_margin,
            indicator.gross_profit_margin,  # 增加毛利率
            indicator.inc_revenue_year_on_year,
            indicator.inc_operation_profit_year_on_year,
            indicator.inc_net_profit_year_on_year,  # 增加利润增长
        ).filter(
            valuation.code.in_(choice),
            valuation.market_cap.between(100, 900),
            valuation.pe_ratio.between(5, 30),  # 增加PE约束
            valuation.pb_ratio.between(1, 10),
            valuation.ps_ratio.between(1, 8),  # 增加PS约束
            valuation.pcf_ratio < 4,
            indicator.eps > 0.2,  # 降低EPS要求
            indicator.roe > 0.15,  # 降低ROE要求，从0.2到0.15
            indicator.net_profit_margin > 0.08,  # 降低净利率要求
            indicator.gross_profit_margin > 0.25,  # 增加毛利率约束
            indicator.inc_revenue_year_on_year > 0.15,  # 降低增长要求
            indicator.inc_operation_profit_year_on_year > 0.08,  # 降低增长要求
            indicator.inc_net_profit_year_on_year > 0.1,  # 增加利润增长约束
        )).set_index('code').index.tolist()

    if len(BM_list) == 0:
        return []

    # 第二层：技术面筛选（30日动量）
    momentum_df = get_price(BM_list, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=31, panel=False, fill_paused=False)
    momentum_df = momentum_df.pivot(index='time', columns='code', values='close')
    # 防止除零错误
    momentum_df_first = momentum_df.iloc[0]
    momentum_df_first[momentum_df_first == 0] = np.nan
    momentum = (momentum_df.iloc[-1] / momentum_df_first - 1) * 100

    # 过滤：30日动量>-5%（允许小幅下跌，避免错过超跌反弹）
    good_momentum_stocks = momentum[momentum > -5].index.tolist()

    # 交叉筛选
    BM_list = [s for s in BM_list if s in good_momentum_stocks]

    if len(BM_list) == 0:
        return []

    # 第三层：综合评分排序（ROE*35% + 增长*35% + 估值*20% + 动量*10%）
    df_fundamentals = get_fundamentals(query(
        valuation.code,
        indicator.roe,
        indicator.inc_revenue_year_on_year,
        indicator.inc_operation_profit_year_on_year,
        indicator.inc_net_profit_year_on_year,
        valuation.pe_ratio,
        valuation.pb_ratio,
    ).filter(valuation.code.in_(BM_list)))

    # 计算综合增长指标
    df_fundamentals['avg_growth'] = (
        df_fundamentals['inc_revenue_year_on_year'] +
        df_fundamentals['inc_operation_profit_year_on_year'] +
        df_fundamentals['inc_net_profit_year_on_year']
    ) / 3

    # 计算估值指标（越低越好）
    df_fundamentals['valuation_score'] = 1 / (df_fundamentals['pe_ratio'] * df_fundamentals['pb_ratio'])

    # 归一化评分
    max_roe = df_fundamentals['roe'].max()
    max_growth = df_fundamentals['avg_growth'].max()
    max_valuation = df_fundamentals['valuation_score'].max()
    max_momentum = momentum.max() - momentum.min()

    df_fundamentals['roe_score'] = df_fundamentals['roe'] / max_roe if max_roe > 0 else 0
    df_fundamentals['growth_score'] = df_fundamentals['avg_growth'] / max_growth if max_growth > 0 else 0
    df_fundamentals['valuation_score'] = df_fundamentals['valuation_score'] / max_valuation if max_valuation > 0 else 0

    momentum_scores = pd.Series(momentum.values, index=momentum.index)
    if max_momentum > 0:
        momentum_scores = (momentum_scores - momentum.min()) / max_momentum

    df_fundamentals['momentum_score'] = df_fundamentals['code'].map(momentum_scores)

    # 综合评分
    df_fundamentals['final_score'] = (
        df_fundamentals['roe_score'] * 0.35 +
        df_fundamentals['growth_score'] * 0.35 +
        df_fundamentals['valuation_score'] * 0.20 +
        df_fundamentals['momentum_score'].fillna(0) * 0.10
    )

    df_fundamentals = df_fundamentals.sort_values('final_score', ascending=False)
    final_list = df_fundamentals['code'].tolist()[:g.stock_num]

    return final_list
# 1-3 整体调整持仓
def monthly_adjustment(context):
    today = context.current_dt
    dt_last = context.previous_date
    N=10  # 原版参数
    B_stocks = get_index_stocks('000300.XSHG', dt_last)
    B_stocks = filter_kcbj_stock(B_stocks)
    B_stocks = filter_st_stock(B_stocks)
    B_stocks = filter_new_stock(context, B_stocks)

    S_stocks = get_index_stocks('399101.XSHE', dt_last)
    S_stocks = filter_kcbj_stock(S_stocks)
    S_stocks = filter_st_stock(S_stocks)
    S_stocks = filter_new_stock(context, S_stocks)

    q = query(
        valuation.code, valuation.circulating_market_cap
    ).filter(
        valuation.code.in_(B_stocks)
    ).order_by(
        valuation.circulating_market_cap.desc()
    )
    df = get_fundamentals(q, date=dt_last)
    Blst = list(df.code)[:20]

    q = query(
        valuation.code, valuation.circulating_market_cap
    ).filter(
        valuation.code.in_(S_stocks)
    ).order_by(
        valuation.circulating_market_cap.asc()
    )
    df = get_fundamentals(q, date=dt_last)
    Slst = list(df.code)[:20]
    #
    B_ratio = get_price(Blst, end_date=dt_last, frequency='1d', fields=['close'], count=N, panel=False
                        ).pivot(index='time', columns='code', values='close')
    change_BIG = (B_ratio.iloc[-1] / B_ratio.iloc[0] - 1) * 100
    A1 = np.array(change_BIG)
    A1 = np.nan_to_num(A1)
    B_mean = np.mean(A1)


    S_ratio = get_price(Slst, end_date=dt_last, frequency='1d', fields=['close'], count=N, panel=False
                        ).pivot(index='time', columns='code', values='close')
    change_SMALL = (S_ratio.iloc[-1] / S_ratio.iloc[0] - 1) * 100
    A1 = np.array(change_SMALL)
    A1 = np.nan_to_num(A1)
    S_mean = np.mean(A1)

    # 波动率过滤已禁用（导致策略收益率大幅下降）
    # # 计算整体波动率
    # total_volatility = np.std(np.concatenate([change_BIG, change_SMALL]))
    #
    # # 波动率过大时，倾向于持有外盘ETF
    # if total_volatility > 25:
    #     print('市场波动率过大，开外盘')
    #     target_list = g.foreign_ETF

    # 无敌好行情：任一风格涨幅超过10%
    if B_mean > 10 or S_mean > 10:
        print('无敌好行情')
        if B_mean > S_mean:
            print('开大')
            choice = B_stocks
            target_list1 = ROIC_BIG(context,choice)
            target_list2 = BIG(context,choice)
            target_list3 = BM(context,choice)
            target_list = target_list3+target_list1+target_list2
            target_list = list(set(target_list))
        else:
            print('开小')
            choice = S_stocks
            target_list = SMALL(context,choice)[:g.stock_num*3]
    # 大盘强且上涨
    elif B_mean>S_mean and B_mean>0:
        print('开大')
        choice = B_stocks
        target_list2 = ROIC_BIG(context,choice)
        target_list1 = BIG(context,choice)
        target_list3 = BM(context,choice)
        target_list = target_list1+target_list2+target_list3
        target_list = list(set(target_list))

    # 小盘强且上涨
    elif B_mean < S_mean and S_mean > 0:
        print('开小')
        choice = S_stocks
        target_list = SMALL(context,choice)[:g.stock_num*3]
    # 其他情况：开外盘
    else:
        print('开外盘')
        target_list = g.foreign_ETF

    # 北向资金过滤（市值替代）- 暂时禁用
    # target_list = add_northbound_filter(context, target_list)

    target_list = filter_limitup_stock(context,target_list)
    target_list = filter_limitdown_stock(context,target_list)
    target_list = filter_paused_stock(target_list)

    # 市场环境过滤：如果是熊市，降低持仓数量或选择外盘 - 暂时禁用
    # is_bear_market = check_market_environment(context)
    # if is_bear_market:
    #     print('检测到熊市环境，降低仓位至50%')
    #     # 熊市时，只买入前g.stock_num//2只股票
    #     target_list = target_list[:g.stock_num//2]

    # 行业分散度过滤：每个行业最多1只股票
    # 暂时禁用（API不稳定，导致策略收益率异常）
    # target_list = filter_industry_diversity(context, target_list, max_per_industry=1)

    # 消除随机性：去重+排序
    target_list = [x for i, x in enumerate(target_list) if x not in target_list[:i]]
    target_list = sorted(target_list)

    for stock in g.hold_list:
        if (stock not in target_list) and (stock not in g.yesterday_HL_list):
            position = context.portfolio.positions[stock]
            close_position(position)
    position_count = len(context.portfolio.positions)
    target_num = len(target_list)
    if target_num > position_count:
        value = context.portfolio.cash / (target_num - position_count)
        for stock in target_list:
            if stock not in list(context.portfolio.positions.keys()):
                if open_position(stock, value):
                    if len(context.portfolio.positions) == target_num:
                        break


# 3-1 交易模块-自定义下单
def order_target_value_(security, value):
    if value == 0:
        log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    return order_target_value(security, value)

# 3-2 交易模块-开仓
def open_position(security, value):
    order = order_target_value_(security, value)
    if order != None and order.filled > 0:
        return True
    return False

# 3-3 交易模块-平仓
def close_position(position):
    security = position.security
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False


def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]

# 2-2 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]


# 2-3 过滤科创北交股票
def filter_kcbj_stock(stock_list):
    return [stock for stock in stock_list
            if stock[0] != '4'
            and stock[0] != '8'
            and stock[:2] != '68'
            and stock[0] != '3']


# 2-4 过滤涨停的股票
def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < current_data[stock].high_limit]


# 2-5 过滤跌停的股票
def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] > current_data[stock].low_limit]


# 2-6 过滤次新股
def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    return [stock for stock in stock_list if
            not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]


