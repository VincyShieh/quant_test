'''
2024.1-2026.1, 收益26%/最大回撤31%
2025.1-2026.1, 53%/12%

原版
102%/15%
64%/12%
'''
# 全天候轮动策略 - 针对性优化版本 v2
# 原策略表现：收益102%, 最大回撤15%
# 优化目标：在保持高收益的基础上，进一步降低回撤，提升夏普比率
# 优化策略：保留原策略核心逻辑，只做微调优化

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import warnings

np.random.seed(42)
warnings.filterwarnings("ignore")

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001, 
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5
    ), type='stock')
    log.set_level('order', 'error')
    
    g.stock_num = 3
    g.hold_list = []
    g.yesterday_HL_list = []
    g.position_tracking = {}  # 新增：持仓追踪
    
    g.foreign_ETF = [
        '518880.XSHG', '513030.XSHG', '513100.XSHG',
        '164824.XSHE', '159866.XSHE',
    ]
    
    run_daily(prepare_stock_list, '9:05')
    run_monthly(monthly_adjustment, 2, '9:30')
    run_daily(stop_loss_enhanced, '14:00')  # 改进止损
    run_daily(take_profit_check, '14:30')   # 新增止盈

def prepare_stock_list(context):
    g.hold_list = list(context.portfolio.positions.keys())
    
    # 更新持仓追踪
    for stock in g.hold_list:
        if stock not in g.position_tracking:
            g.position_tracking[stock] = {
                'highest_price': context.portfolio.positions[stock].price,
                'buy_price': context.portfolio.positions[stock].avg_cost
            }
        else:
            current_price = context.portfolio.positions[stock].price
            if current_price > g.position_tracking[stock]['highest_price']:
                g.position_tracking[stock]['highest_price'] = current_price
    
    g.yesterday_HL_list = []
    if g.hold_list:
        df = get_price(g.hold_list, end_date=context.previous_date, 
                      frequency='daily', fields=['close', 'high_limit'],
                      count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)

def stop_loss_enhanced(context):
    """改进止损：保留原逻辑 + 移动止损"""
    now_time = context.current_dt
    num = 0
    
    # 1. 处理昨日涨停（原逻辑）
    if g.yesterday_HL_list:
        for stock in g.yesterday_HL_list:
            if stock not in context.portfolio.positions:
                continue
            current_data = get_price(stock, end_date=now_time, frequency='1m', 
                                   fields=['close', 'high_limit'],
                                   skip_paused=False, fq='pre', count=1, 
                                   panel=False, fill_paused=True)
            if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                log.info(f"[{stock}] 涨停打开")
                close_position(context.portfolio.positions[stock])
                if stock in g.position_tracking:
                    del g.position_tracking[stock]
                num += 1
    
    # 2. 改进的止损逻辑
    SS = []
    S = []
    for stock in g.hold_list:
        if stock not in context.portfolio.positions:
            continue
            
        position = context.portfolio.positions[stock]
        current_price = position.price
        avg_cost = position.avg_cost
        return_pct = (current_price - avg_cost) / avg_cost
        
        should_sell = False
        
        # 基础止损：-8%（保持原逻辑）
        if return_pct < -0.08:
            should_sell = True
            log.info(f"[{stock}] 止损 {return_pct:.2%}")
        
        # 移动止损：盈利>5%后，回撤>6%
        elif stock in g.position_tracking and return_pct > 0.05:
            highest = g.position_tracking[stock]['highest_price']
            drawdown = (current_price - highest) / highest
            if drawdown < -0.06:
                should_sell = True
                log.info(f"[{stock}] 移动止损 {drawdown:.2%}")
        
        if should_sell:
            order_target_value(stock, 0)
            if stock in g.position_tracking:
                del g.position_tracking[stock]
            num += 1
        else:
            S.append(stock)
            SS.append(return_pct)
    
    # 3. 补仓（原逻辑）
    if num >= 1 and len(SS) > 0:
        num = min(3, len(S))
        min_values = sorted(SS)[:num]
        min_indices = [SS.index(value) for value in min_values]
        min_stocks = [S[index] for index in min_indices]
        
        cash = context.portfolio.cash / num
        for stock in min_stocks:
            order_value(stock, cash)

def take_profit_check(context):
    """新增：20%止盈"""
    for stock in g.hold_list:
        if stock not in context.portfolio.positions:
            continue
        
        position = context.portfolio.positions[stock]
        return_pct = (position.price - position.avg_cost) / position.avg_cost
        
        if return_pct > 0.20:
            log.info(f"[{stock}] 止盈 {return_pct:.2%}")
            close_position(position)
            if stock in g.position_tracking:
                del g.position_tracking[stock]

# ==================== 选股函数（适度放宽） ====================
def filter_roic(context, stock_list):
    yesterday = context.previous_date
    result = []
    for stock in stock_list:
        try:
            roic = get_factor_values(stock, 'roic_ttm', end_date=yesterday, count=1)['roic_ttm'].iloc[0, 0]
            if roic and roic > 0.06:
                result.append(stock)
        except:
            continue
    return result

def SMALL(context, choice):
    df = get_fundamentals(query(
        valuation.code, indicator.roe, indicator.roa,
        valuation.pe_ratio, valuation.pb_ratio,
        indicator.inc_revenue_year_on_year,
        indicator.inc_net_profit_year_on_year,
        valuation.turnover_ratio,
    ).filter(
        valuation.code.in_(choice),
        indicator.roe > 0.10,
        indicator.roa > 0.06,
        valuation.pe_ratio.between(5, 60),
        valuation.pb_ratio.between(0.8, 10),
        indicator.inc_revenue_year_on_year > 0.08,
        indicator.inc_net_profit_year_on_year > 0.08,
        valuation.turnover_ratio > 2,
    )).set_index('code').index.tolist()

    if not df:
        return []

    momentum_df = get_price(df, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=21, panel=False, fill_paused=False)
    momentum_df = momentum_df.pivot(index='time', columns='code', values='close')
    momentum = (momentum_df.iloc[-1] / momentum_df.iloc[0] - 1) * 100

    good_stocks = momentum[momentum > -2].index.tolist()
    df = [s for s in df if s in good_stocks]

    if not df:
        return []

    q = query(valuation.code).filter(valuation.code.in_(df)).order_by(valuation.market_cap.asc())
    return list(get_fundamentals(q).code)[:g.stock_num * 3]

def BIG(context, choice):
    df = get_fundamentals(query(
        valuation.code, valuation.pe_ratio_lyr, valuation.ps_ratio,
        valuation.pcf_ratio, indicator.eps, indicator.roe,
        indicator.net_profit_margin, indicator.gross_profit_margin,
        indicator.inc_revenue_year_on_year,
        balance.total_liability, balance.total_sheet_owner_equities,
    ).filter(
        valuation.code.in_(choice),
        valuation.pe_ratio_lyr.between(5, 35),
        valuation.ps_ratio.between(1, 10),
        valuation.pcf_ratio < 12,
        indicator.eps > 0.25,
        indicator.roe > 0.08,
        indicator.net_profit_margin > 0.08,
        indicator.gross_profit_margin > 0.25,
        indicator.inc_revenue_year_on_year > 0.12,
        (balance.total_liability / balance.total_sheet_owner_equities) < 1.8,
    )).set_index('code').index.tolist()

    if not df:
        return []

    momentum_df = get_price(df, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=61, panel=False, fill_paused=False)
    momentum_df = momentum_df.pivot(index='time', columns='code', values='close')
    momentum = (momentum_df.iloc[-1] / momentum_df.iloc[0] - 1) * 100

    good_stocks = momentum[momentum > 3].index.tolist()
    df = [s for s in df if s in good_stocks]

    if not df:
        return []

    df_fundamentals = get_fundamentals(query(
        valuation.code, indicator.roe,
    ).filter(valuation.code.in_(df)))

    max_roe = df_fundamentals['roe'].max()
    df_fundamentals['score'] = (
        df_fundamentals['roe'] / max_roe * 0.6 +
        (momentum[df_fundamentals['code']].values / momentum.max() * 0.4)
    )

    df_fundamentals = df_fundamentals.sort_values('score', ascending=False)
    return df_fundamentals['code'].tolist()[:g.stock_num]

def ROIC_BIG(context, choice):
    df = get_fundamentals(query(
        valuation.code,
    ).filter(
        valuation.code.in_(choice),
        valuation.market_cap > 150,
        valuation.pe_ratio.between(0, 60),
        indicator.eps > 0.08,
        indicator.roa > 0.10,
        (balance.total_liability / balance.total_sheet_owner_equities) < 0.8,
        indicator.inc_total_revenue_year_on_year > 0.15,
        indicator.inc_revenue_year_on_year > 0.12,
        balance.retained_profit > 0,
    )).set_index('code').index.tolist()

    if not df:
        return []

    df = filter_roic(context, df)
    if not df:
        return []

    df_fundamentals = get_fundamentals(query(
        valuation.code, indicator.roe,
    ).filter(valuation.code.in_(df)))

    roic_dict = {}
    for stock in df:
        try:
            roic = get_factor_values(stock, 'roic_ttm', end_date=context.previous_date, count=1)['roic_ttm'].iloc[0, 0]
            roic_dict[stock] = roic if roic else 0
        except:
            roic_dict[stock] = 0

    momentum_df = get_price(df, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=61, panel=False, fill_paused=False)
    momentum_df = momentum_df.pivot(index='time', columns='code', values='close')
    momentum = (momentum_df.iloc[-1] / momentum_df.iloc[0] - 1) * 100

    score_df = pd.DataFrame({'code': df})
    score_df['roic'] = score_df['code'].map(roic_dict)
    score_df['roe'] = score_df['code'].map(df_fundamentals.set_index('code')['roe'])
    score_df['momentum'] = score_df['code'].map(momentum)

    max_roic = score_df['roic'].max()
    max_roe = score_df['roe'].max()
    max_momentum = score_df['momentum'].max() - score_df['momentum'].min()

    score_df['roic_score'] = score_df['roic'] / max_roic if max_roic > 0 else 0
    score_df['roe_score'] = score_df['roe'] / max_roe if max_roe > 0 else 0
    score_df['momentum_score'] = (score_df['momentum'] - score_df['momentum'].min()) / max_momentum if max_momentum > 0 else 0

    score_df['final_score'] = (
        score_df['roic_score'] * 0.5 +
        score_df['roe_score'] * 0.3 +
        score_df['momentum_score'] * 0.2
    )

    score_df = score_df.sort_values('final_score', ascending=False)
    return score_df['code'].tolist()[:g.stock_num]

def BM(context, choice):
    BM_list = get_fundamentals(query(
        valuation.code, valuation.pe_ratio, valuation.ps_ratio,
        valuation.pcf_ratio, indicator.eps, indicator.roe,
        indicator.net_profit_margin, indicator.gross_profit_margin,
        indicator.inc_revenue_year_on_year,
        indicator.inc_operation_profit_year_on_year,
        indicator.inc_net_profit_year_on_year,
    ).filter(
        valuation.code.in_(choice),
        valuation.market_cap.between(80, 1000),
        valuation.pe_ratio.between(5, 40),
        valuation.pb_ratio.between(0.8, 12),
        valuation.ps_ratio.between(0.8, 10),
        valuation.pcf_ratio < 5,
        indicator.eps > 0.15,
        indicator.roe > 0.12,
        indicator.net_profit_margin > 0.06,
        indicator.gross_profit_margin > 0.20,
        indicator.inc_revenue_year_on_year > 0.12,
        indicator.inc_operation_profit_year_on_year > 0.06,
        indicator.inc_net_profit_year_on_year > 0.08,
    )).set_index('code').index.tolist()

    if not BM_list:
        return []

    momentum_df = get_price(BM_list, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=31, panel=False, fill_paused=False)
    momentum_df = momentum_df.pivot(index='time', columns='code', values='close')
    momentum = (momentum_df.iloc[-1] / momentum_df.iloc[0] - 1) * 100

    good_stocks = momentum[momentum > -5].index.tolist()
    BM_list = [s for s in BM_list if s in good_stocks]

    if not BM_list:
        return []

    df_fundamentals = get_fundamentals(query(
        valuation.code, indicator.roe,
        indicator.inc_revenue_year_on_year,
        indicator.inc_operation_profit_year_on_year,
        indicator.inc_net_profit_year_on_year,
        valuation.pe_ratio, valuation.pb_ratio,
    ).filter(valuation.code.in_(BM_list)))

    df_fundamentals['avg_growth'] = (
        df_fundamentals['inc_revenue_year_on_year'] +
        df_fundamentals['inc_operation_profit_year_on_year'] +
        df_fundamentals['inc_net_profit_year_on_year']
    ) / 3

    df_fundamentals['valuation_score'] = 1 / (df_fundamentals['pe_ratio'] * df_fundamentals['pb_ratio'])

    max_roe = df_fundamentals['roe'].max()
    max_growth = df_fundamentals['avg_growth'].max()
    max_valuation = df_fundamentals['valuation_score'].max()
    max_momentum = momentum.max() - momentum.min()

    df_fundamentals['roe_score'] = df_fundamentals['roe'] / max_roe if max_roe > 0 else 0
    df_fundamentals['growth_score'] = df_fundamentals['avg_growth'] / max_growth if max_growth > 0 else 0
    df_fundamentals['val_score'] = df_fundamentals['valuation_score'] / max_valuation if max_valuation > 0 else 0

    momentum_scores = pd.Series(momentum.values, index=momentum.index)
    if max_momentum > 0:
        momentum_scores = (momentum_scores - momentum.min()) / max_momentum

    df_fundamentals['momentum_score'] = df_fundamentals['code'].map(momentum_scores)

    df_fundamentals['final_score'] = (
        df_fundamentals['roe_score'] * 0.35 +
        df_fundamentals['growth_score'] * 0.35 +
        df_fundamentals['val_score'] * 0.20 +
        df_fundamentals['momentum_score'].fillna(0) * 0.10
    )

    df_fundamentals = df_fundamentals.sort_values('final_score', ascending=False)
    return df_fundamentals['code'].tolist()[:g.stock_num]

# ==================== 月度调仓 ====================
def monthly_adjustment(context):
    dt_last = context.previous_date
    N = 10
    
    B_stocks = get_index_stocks('000300.XSHG', dt_last)
    B_stocks = filter_kcbj_stock(B_stocks)
    B_stocks = filter_st_stock(B_stocks)
    B_stocks = filter_new_stock(context, B_stocks)
    
    S_stocks = get_index_stocks('399101.XSHE', dt_last)
    S_stocks = filter_kcbj_stock(S_stocks)
    S_stocks = filter_st_stock(S_stocks)
    S_stocks = filter_new_stock(context, S_stocks)
    
    q = query(valuation.code, valuation.circulating_market_cap).filter(
        valuation.code.in_(B_stocks)
    ).order_by(valuation.circulating_market_cap.desc())
    df = get_fundamentals(q, date=dt_last)
    Blst = list(df.code)[:20]
    
    q = query(valuation.code, valuation.circulating_market_cap).filter(
        valuation.code.in_(S_stocks)
    ).order_by(valuation.circulating_market_cap.asc())
    df = get_fundamentals(q, date=dt_last)
    Slst = list(df.code)[:20]
    
    B_ratio = get_price(Blst, end_date=dt_last, frequency='1d', fields=['close'], 
                       count=N, panel=False).pivot(index='time', columns='code', values='close')
    change_BIG = (B_ratio.iloc[-1] / B_ratio.iloc[0] - 1) * 100
    B_mean = np.mean(np.nan_to_num(np.array(change_BIG)))
    
    S_ratio = get_price(Slst, end_date=dt_last, frequency='1d', fields=['close'], 
                       count=N, panel=False).pivot(index='time', columns='code', values='close')
    change_SMALL = (S_ratio.iloc[-1] / S_ratio.iloc[0] - 1) * 100
    S_mean = np.mean(np.nan_to_num(np.array(change_SMALL)))
    
    # 选股（保留原逻辑）
    if B_mean > 10 or S_mean > 10:
        if B_mean > S_mean:
            choice = B_stocks
            target_list = BM(context, choice) + ROIC_BIG(context, choice) + BIG(context, choice)
            target_list = list(set(target_list))
        else:
            choice = S_stocks
            target_list = SMALL(context, choice)[:g.stock_num * 3]
    elif B_mean > S_mean and B_mean > 0:
        choice = B_stocks
        target_list = BIG(context, choice) + ROIC_BIG(context, choice) + BM(context, choice)
        target_list = list(set(target_list))
    elif B_mean < S_mean and S_mean > 0:
        choice = S_stocks
        target_list = SMALL(context, choice)[:g.stock_num * 3]
    else:
        target_list = g.foreign_ETF
    
    target_list = filter_limitup_stock(context, target_list)
    target_list = filter_limitdown_stock(context, target_list)
    target_list = filter_paused_stock(target_list)
    
    # 确保确定性
    target_list = [x for i, x in enumerate(target_list) if x not in target_list[:i]]
    target_list = sorted(target_list)
    
    # 调仓
    for stock in g.hold_list:
        if (stock not in target_list) and (stock not in g.yesterday_HL_list):
            close_position(context.portfolio.positions[stock])
            if stock in g.position_tracking:
                del g.position_tracking[stock]
    
    position_count = len(context.portfolio.positions)
    target_num = len(target_list)
    
    if target_num > position_count:
        value = context.portfolio.cash / (target_num - position_count)
        for stock in target_list:
            if stock not in context.portfolio.positions:
                if open_position(stock, value):
                    if len(context.portfolio.positions) == target_num:
                        break

def order_target_value_(security, value):
    return order_target_value(security, value)

def open_position(security, value):
    order = order_target_value_(security, value)
    return order and order.filled > 0

def close_position(position):
    security = position.security
    order = order_target_value_(security, 0)
    return order and order.status == OrderStatus.held and order.filled == order.amount

def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]

def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]

def filter_kcbj_stock(stock_list):
    return [stock for stock in stock_list 
            if stock[0] not in ['4', '8'] and stock[:2] != '68' and stock[0] != '3']

def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list 
            if stock in context.portfolio.positions
            or last_prices[stock][-1] < current_data[stock].high_limit]

def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list 
            if stock in context.portfolio.positions
            or last_prices[stock][-1] > current_data[stock].low_limit]

def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    return [stock for stock in stock_list 
            if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]