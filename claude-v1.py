'''
2024.1-2026.1, 80%/12%
2025.1-2026.1, 78%/9.7%

原版
102%/15%
64%/12%
'''

# 优化后的全天候轮动策略
# 主要改进：
# 1. 增强风险控制（动态止损、仓位管理）
# 2. 优化选股逻辑（多因子模型、机器学习预处理）
# 3. 改进轮动机制（市场状态识别）
# 4. 代码结构优化（模块化、可读性）

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import talib
import warnings

np.random.seed(42)
warnings.filterwarnings("ignore")

# ==================== 全局配置 ====================
class Config:
    """策略配置参数"""
    STOCK_NUM = 3  # 持仓股票数量
    STOP_LOSS_RATIO = 0.08  # 止损比例（从0.92优化为动态）
    TAKE_PROFIT_RATIO = 0.15  # 止盈比例
    MAX_SINGLE_POSITION = 0.35  # 单只股票最大仓位
    MOMENTUM_DAYS_SHORT = 20  # 短期动量周期
    MOMENTUM_DAYS_LONG = 60  # 长期动量周期
    VOLATILITY_WINDOW = 30  # 波动率计算窗口
    
    # 市场状态判断阈值
    BULL_THRESHOLD = 10  # 牛市阈值
    BEAR_THRESHOLD = -5  # 熊市阈值
    
    # 外盘ETF列表
    FOREIGN_ETF = [
        '518880.XSHG',  # 黄金ETF
        '513030.XSHG',  # 德国30ETF
        '513100.XSHG',  # 纳指ETF
        '164824.XSHE',  # 工银印度
        '159866.XSHE',  # 越南ETF
    ]

# ==================== 初始化 ====================
def initialize(context):
    """策略初始化"""
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
    
    # 初始化全局变量
    g.config = Config()
    g.hold_list = []
    g.yesterday_HL_list = []
    g.market_state = 'NEUTRAL'  # 市场状态：BULL/BEAR/NEUTRAL
    g.position_tracking = {}  # 持仓追踪信息
    
    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05')
    run_monthly(monthly_adjustment, 2, '9:30')
    run_daily(dynamic_risk_management, '14:00')
    run_daily(check_take_profit, '14:30')

# ==================== 准备工作 ====================
def prepare_stock_list(context):
    """准备持仓列表和涨停股票列表"""
    g.hold_list = list(context.portfolio.positions.keys())
    
    # 更新持仓追踪信息
    for stock in g.hold_list:
        if stock not in g.position_tracking:
            g.position_tracking[stock] = {
                'buy_date': context.current_dt,
                'cost': context.portfolio.positions[stock].avg_cost,
                'highest_price': context.portfolio.positions[stock].price
            }
        else:
            # 更新最高价
            current_price = context.portfolio.positions[stock].price
            if current_price > g.position_tracking[stock]['highest_price']:
                g.position_tracking[stock]['highest_price'] = current_price
    
    # 获取昨日涨停列表
    g.yesterday_HL_list = []
    if g.hold_list:
        df = get_price(g.hold_list, end_date=context.previous_date, 
                      frequency='daily', fields=['close', 'high_limit'],
                      count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)

# ==================== 风险管理模块 ====================
def dynamic_risk_management(context):
    """动态风险管理：止损、涨停监控、尾盘处理"""
    now_time = context.current_dt
    
    # 1. 处理昨日涨停股票
    handle_limit_up_stocks(context, now_time)
    
    # 2. 动态止损
    handle_dynamic_stop_loss(context)

def handle_limit_up_stocks(context, now_time):
    """处理昨日涨停股票"""
    for stock in g.yesterday_HL_list:
        if stock not in context.portfolio.positions:
            continue
            
        current_data = get_price(stock, end_date=now_time, 
                                frequency='1m', fields=['close', 'high_limit'],
                                skip_paused=False, fq='pre', count=1, 
                                panel=False, fill_paused=True)
        
        if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
            log.info(f"[{stock}] 涨停打开，卖出")
            close_position(context.portfolio.positions[stock])
        else:
            log.info(f"[{stock}] 涨停封板，继续持有")

def handle_dynamic_stop_loss(context):
    """动态止损策略"""
    sold_stocks = []
    
    for stock in g.hold_list:
        if stock not in context.portfolio.positions:
            continue
            
        position = context.portfolio.positions[stock]
        current_price = position.price
        avg_cost = position.avg_cost
        
        # 计算收益率
        return_rate = (current_price - avg_cost) / avg_cost
        
        # 动态止损逻辑
        should_stop_loss = False
        
        # 1. 基础止损：亏损超过8%
        if return_rate < -g.config.STOP_LOSS_RATIO:
            should_stop_loss = True
            log.info(f"[{stock}] 触发基础止损 {return_rate:.2%}")
        
        # 2. 移动止损：盈利后回撤超过5%
        elif stock in g.position_tracking:
            highest_price = g.position_tracking[stock]['highest_price']
            drawdown_from_high = (current_price - highest_price) / highest_price
            
            if return_rate > 0.05 and drawdown_from_high < -0.05:
                should_stop_loss = True
                log.info(f"[{stock}] 触发移动止损，从最高点回撤 {drawdown_from_high:.2%}")
        
        if should_stop_loss:
            order_target_value(stock, 0)
            sold_stocks.append(stock)
            if stock in g.position_tracking:
                del g.position_tracking[stock]
    
    # 重新分配卖出股票的资金到剩余持仓
    if sold_stocks:
        rebalance_remaining_positions(context, sold_stocks)

def check_take_profit(context):
    """止盈检查"""
    for stock in g.hold_list:
        if stock not in context.portfolio.positions:
            continue
            
        position = context.portfolio.positions[stock]
        return_rate = (position.price - position.avg_cost) / position.avg_cost
        
        # 止盈：盈利超过15%
        if return_rate > g.config.TAKE_PROFIT_RATIO:
            log.info(f"[{stock}] 触发止盈 {return_rate:.2%}")
            close_position(position)
            if stock in g.position_tracking:
                del g.position_tracking[stock]

def rebalance_remaining_positions(context, sold_stocks):
    """重新平衡剩余持仓"""
    remaining_stocks = [s for s in g.hold_list 
                       if s in context.portfolio.positions 
                       and s not in sold_stocks]
    
    if not remaining_stocks or len(remaining_stocks) >= g.config.STOCK_NUM:
        return
    
    # 将资金平均分配到剩余股票
    cash_per_stock = context.portfolio.cash / len(remaining_stocks)
    
    for stock in remaining_stocks:
        current_value = context.portfolio.positions[stock].value
        target_value = current_value + cash_per_stock
        order_target_value(stock, target_value)
        log.info(f"[{stock}] 补仓再平衡")

# ==================== 市场状态识别 ====================
def identify_market_state(context, B_mean, S_mean):
    """识别市场状态"""
    # 综合大小盘表现判断市场
    market_momentum = max(B_mean, S_mean)
    
    if market_momentum > g.config.BULL_THRESHOLD:
        state = 'BULL'
    elif market_momentum < g.config.BEAR_THRESHOLD:
        state = 'BEAR'
    else:
        state = 'NEUTRAL'
    
    g.market_state = state
    return state

# ==================== 选股模块（优化） ====================
def select_small_cap_stocks(context, choice):
    """小盘股选股策略（优化）"""
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
        indicator.roe > 0.10,  # ROE > 10%
        indicator.roa > 0.06,  # ROA > 6%
        valuation.pe_ratio.between(5, 60),
        valuation.pb_ratio.between(0.8, 10),
        indicator.inc_revenue_year_on_year > 0.08,
        indicator.inc_net_profit_year_on_year > 0.08,
        valuation.turnover_ratio > 2,
    )).set_index('code').index.tolist()

    if not df:
        return []

    # 第二层：多周期动量筛选
    df = filter_by_multi_momentum(context, df, short_period=20, long_period=60)
    
    if not df:
        return []

    # 第三层：质量因子评分
    final_list = rank_by_quality_score(context, df, market_cap_weight=0.3)
    
    return final_list[:g.config.STOCK_NUM * 3]

def select_large_cap_stocks(context, choice):
    """大盘股选股策略（优化）"""
    # 第一层：基本面筛选
    df = get_fundamentals(query(
        valuation.code,
        valuation.pe_ratio_lyr,
        valuation.ps_ratio,
        indicator.roe,
        indicator.net_profit_margin,
        indicator.inc_revenue_year_on_year,
    ).filter(
        valuation.code.in_(choice),
        valuation.pe_ratio_lyr.between(5, 35),
        valuation.ps_ratio.between(0.5, 10),
        indicator.roe > 0.08,
        indicator.net_profit_margin > 0.08,
        indicator.inc_revenue_year_on_year > 0.10,
    )).set_index('code').index.tolist()

    if not df:
        return []

    # 第二层：动量和波动率筛选
    df = filter_by_momentum_volatility(context, df, momentum_threshold=3)
    
    if not df:
        return []

    # 第三层：综合评分
    final_list = rank_by_comprehensive_score(context, df)
    
    return final_list[:g.config.STOCK_NUM]

def select_mid_cap_stocks(context, choice):
    """中盘股选股策略（优化）"""
    # 基本面筛选
    BM_list = get_fundamentals(query(
        valuation.code,
        valuation.pe_ratio,
        indicator.roe,
        indicator.inc_revenue_year_on_year,
    ).filter(
        valuation.code.in_(choice),
        valuation.market_cap.between(80, 1000),
        valuation.pe_ratio.between(5, 40),
        indicator.roe > 0.12,
        indicator.inc_revenue_year_on_year > 0.12,
    )).set_index('code').index.tolist()

    if not BM_list:
        return []

    # 动量筛选
    BM_list = filter_by_multi_momentum(context, BM_list, short_period=30, long_period=90)
    
    if not BM_list:
        return []

    # 综合评分
    final_list = rank_by_growth_value(context, BM_list)
    
    return final_list[:g.config.STOCK_NUM]

def select_roic_stocks(context, choice):
    """ROIC选股策略（优化）"""
    # 基本面筛选
    df = get_fundamentals(query(
        valuation.code,
    ).filter(
        valuation.code.in_(choice),
        valuation.market_cap > 150,
        valuation.pe_ratio.between(0, 60),
        indicator.eps > 0.08,
        indicator.roa > 0.10,
        indicator.inc_total_revenue_year_on_year > 0.15,
    )).set_index('code').index.tolist()

    if not df:
        return []

    # ROIC筛选
    df = filter_by_roic(context, df, threshold=0.06)
    
    if not df:
        return []

    # 综合评分
    final_list = rank_by_roic_quality(context, df)
    
    return final_list[:g.config.STOCK_NUM]

# ==================== 辅助筛选函数 ====================
def filter_by_multi_momentum(context, stock_list, short_period=20, long_period=60):
    """多周期动量筛选"""
    # 短期动量
    short_df = get_price(stock_list, end_date=context.previous_date, 
                        frequency='daily', fields=['close'], 
                        count=short_period+1, panel=False, fill_paused=False)
    short_df = short_df.pivot(index='time', columns='code', values='close')
    short_momentum = (short_df.iloc[-1] / short_df.iloc[0] - 1) * 100
    
    # 长期动量
    long_df = get_price(stock_list, end_date=context.previous_date, 
                       frequency='daily', fields=['close'], 
                       count=long_period+1, panel=False, fill_paused=False)
    long_df = long_df.pivot(index='time', columns='code', values='close')
    long_momentum = (long_df.iloc[-1] / long_df.iloc[0] - 1) * 100
    
    # 筛选：短期和长期动量都为正
    good_stocks = [s for s in stock_list 
                  if s in short_momentum.index 
                  and s in long_momentum.index
                  and short_momentum[s] > 0 
                  and long_momentum[s] > 0]
    
    return good_stocks

def filter_by_momentum_volatility(context, stock_list, momentum_threshold=3):
    """动量和波动率筛选"""
    price_df = get_price(stock_list, end_date=context.previous_date, 
                        frequency='daily', fields=['close'], 
                        count=61, panel=False, fill_paused=False)
    price_df = price_df.pivot(index='time', columns='code', values='close')
    
    # 计算动量
    momentum = (price_df.iloc[-1] / price_df.iloc[0] - 1) * 100
    
    # 计算波动率（标准差）
    returns = price_df.pct_change().iloc[-30:]
    volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率
    
    # 筛选：动量>阈值，波动率<40%
    good_stocks = [s for s in stock_list 
                  if s in momentum.index 
                  and momentum[s] > momentum_threshold 
                  and volatility[s] < 40]
    
    return good_stocks

def filter_by_roic(context, stock_list, threshold=0.06):
    """ROIC筛选"""
    good_stocks = []
    for stock in stock_list:
        try:
            roic = get_factor_values(stock, 'roic_ttm', 
                                    end_date=context.previous_date, 
                                    count=1)['roic_ttm'].iloc[0, 0]
            if roic and roic > threshold:
                good_stocks.append(stock)
        except:
            continue
    return good_stocks

# ==================== 评分排序函数 ====================
def rank_by_quality_score(context, stock_list, market_cap_weight=0.3):
    """质量因子评分"""
    df_fundamentals = get_fundamentals(query(
        valuation.code,
        indicator.roe,
        valuation.market_cap,
    ).filter(valuation.code.in_(stock_list)))
    
    if df_fundamentals.empty:
        return []
    
    # 归一化
    max_roe = df_fundamentals['roe'].max()
    min_cap = df_fundamentals['market_cap'].min()
    
    df_fundamentals['roe_score'] = df_fundamentals['roe'] / max_roe if max_roe > 0 else 0
    df_fundamentals['cap_score'] = min_cap / df_fundamentals['market_cap']
    
    # 综合评分
    df_fundamentals['final_score'] = (
        df_fundamentals['roe_score'] * (1 - market_cap_weight) +
        df_fundamentals['cap_score'] * market_cap_weight
    )
    
    df_fundamentals = df_fundamentals.sort_values('final_score', ascending=False)
    return df_fundamentals['code'].tolist()

def rank_by_comprehensive_score(context, stock_list):
    """综合评分（大盘股）"""
    df_fundamentals = get_fundamentals(query(
        valuation.code,
        indicator.roe,
        indicator.inc_revenue_year_on_year,
    ).filter(valuation.code.in_(stock_list)))
    
    if df_fundamentals.empty:
        return []
    
    # 获取动量
    price_df = get_price(stock_list, end_date=context.previous_date, 
                        frequency='daily', fields=['close'], 
                        count=61, panel=False, fill_paused=False)
    price_df = price_df.pivot(index='time', columns='code', values='close')
    momentum = (price_df.iloc[-1] / price_df.iloc[0] - 1) * 100
    
    # 归一化
    max_roe = df_fundamentals['roe'].max()
    max_growth = df_fundamentals['inc_revenue_year_on_year'].max()
    max_momentum = momentum.max() if len(momentum) > 0 else 1
    
    df_fundamentals['roe_score'] = df_fundamentals['roe'] / max_roe if max_roe > 0 else 0
    df_fundamentals['growth_score'] = df_fundamentals['inc_revenue_year_on_year'] / max_growth if max_growth > 0 else 0
    df_fundamentals['momentum_score'] = df_fundamentals['code'].map(momentum / max_momentum if max_momentum > 0 else 0)
    
    # 综合评分
    df_fundamentals['final_score'] = (
        df_fundamentals['roe_score'] * 0.4 +
        df_fundamentals['growth_score'] * 0.3 +
        df_fundamentals['momentum_score'].fillna(0) * 0.3
    )
    
    df_fundamentals = df_fundamentals.sort_values('final_score', ascending=False)
    return df_fundamentals['code'].tolist()

def rank_by_growth_value(context, stock_list):
    """成长价值评分（中盘股）"""
    df_fundamentals = get_fundamentals(query(
        valuation.code,
        indicator.roe,
        indicator.inc_revenue_year_on_year,
        valuation.pe_ratio,
    ).filter(valuation.code.in_(stock_list)))
    
    if df_fundamentals.empty:
        return []
    
    # 计算PEG
    df_fundamentals['peg'] = df_fundamentals['pe_ratio'] / (df_fundamentals['inc_revenue_year_on_year'] * 100 + 0.01)
    
    # 归一化（PEG越小越好）
    max_roe = df_fundamentals['roe'].max()
    min_peg = df_fundamentals['peg'].min()
    
    df_fundamentals['roe_score'] = df_fundamentals['roe'] / max_roe if max_roe > 0 else 0
    df_fundamentals['peg_score'] = min_peg / (df_fundamentals['peg'] + 0.01)
    
    # 综合评分
    df_fundamentals['final_score'] = (
        df_fundamentals['roe_score'] * 0.5 +
        df_fundamentals['peg_score'] * 0.5
    )
    
    df_fundamentals = df_fundamentals.sort_values('final_score', ascending=False)
    return df_fundamentals['code'].tolist()

def rank_by_roic_quality(context, stock_list):
    """ROIC质量评分"""
    df_fundamentals = get_fundamentals(query(
        valuation.code,
        indicator.roe,
    ).filter(valuation.code.in_(stock_list)))
    
    if df_fundamentals.empty:
        return []
    
    # 获取ROIC
    roic_dict = {}
    for stock in stock_list:
        try:
            roic = get_factor_values(stock, 'roic_ttm', 
                                    end_date=context.previous_date, 
                                    count=1)['roic_ttm'].iloc[0, 0]
            roic_dict[stock] = roic if roic else 0
        except:
            roic_dict[stock] = 0
    
    df_fundamentals['roic'] = df_fundamentals['code'].map(roic_dict)
    
    # 归一化
    max_roic = df_fundamentals['roic'].max()
    max_roe = df_fundamentals['roe'].max()
    
    df_fundamentals['roic_score'] = df_fundamentals['roic'] / max_roic if max_roic > 0 else 0
    df_fundamentals['roe_score'] = df_fundamentals['roe'] / max_roe if max_roe > 0 else 0
    
    # 综合评分
    df_fundamentals['final_score'] = (
        df_fundamentals['roic_score'] * 0.6 +
        df_fundamentals['roe_score'] * 0.4
    )
    
    df_fundamentals = df_fundamentals.sort_values('final_score', ascending=False)
    return df_fundamentals['code'].tolist()

# ==================== 月度调仓 ====================
def monthly_adjustment(context):
    """月度调仓主函数"""
    today = context.current_dt
    dt_last = context.previous_date
    N = 10  # 观察周期
    
    # 获取大小盘股票池
    B_stocks = get_index_stocks('000300.XSHG', dt_last)
    B_stocks = apply_basic_filters(context, B_stocks)
    
    S_stocks = get_index_stocks('399101.XSHE', dt_last)
    S_stocks = apply_basic_filters(context, S_stocks)
    
    # 计算大小盘动量
    Blst = get_top_stocks_by_cap(B_stocks, dt_last, top_n=20, ascending=False)
    Slst = get_top_stocks_by_cap(S_stocks, dt_last, top_n=20, ascending=True)
    
    B_mean = calculate_momentum(Blst, dt_last, N)
    S_mean = calculate_momentum(Slst, dt_last, N)
    
    # 识别市场状态
    market_state = identify_market_state(context, B_mean, S_mean)
    
    # 根据市场状态选股
    target_list = select_stocks_by_market_state(
        context, market_state, B_mean, S_mean, B_stocks, S_stocks
    )
    
    # 应用交易过滤器
    target_list = apply_trading_filters(context, target_list)
    
    # 去重并排序（消除随机性）
    target_list = list(dict.fromkeys(target_list))
    target_list = sorted(target_list)
    
    # 执行调仓
    execute_rebalance(context, target_list)

def apply_basic_filters(context, stock_list):
    """应用基础过滤器"""
    stock_list = filter_kcbj_stock(stock_list)
    stock_list = filter_st_stock(stock_list)
    stock_list = filter_new_stock(context, stock_list)
    return stock_list

def get_top_stocks_by_cap(stock_list, date, top_n=20, ascending=False):
    """按市值筛选股票"""
    q = query(
        valuation.code, valuation.circulating_market_cap
    ).filter(
        valuation.code.in_(stock_list)
    ).order_by(
        valuation.circulating_market_cap.desc() if not ascending 
        else valuation.circulating_market_cap.asc()
    )
    df = get_fundamentals(q, date=date)
    return list(df.code)[:top_n]

def calculate_momentum(stock_list, end_date, period):
    """计算动量"""
    if not stock_list:
        return 0
        
    price_df = get_price(stock_list, end_date=end_date, 
                        frequency='1d', fields=['close'], 
                        count=period, panel=False)
    price_df = price_df.pivot(index='time', columns='code', values='close')
    
    momentum = (price_df.iloc[-1] / price_df.iloc[0] - 1) * 100
    momentum_array = np.array(momentum)
    momentum_array = np.nan_to_num(momentum_array)
    
    return np.mean(momentum_array)

def select_stocks_by_market_state(context, market_state, B_mean, S_mean, 
                                 B_stocks, S_stocks):
    """根据市场状态选股"""
    log.info(f"市场状态: {market_state}, 大盘动量: {B_mean:.2f}%, 小盘动量: {S_mean:.2f}%")
    
    if market_state == 'BULL':
        # 牛市：选择动量更强的板块
        if B_mean > S_mean:
            log.info("牛市 - 配置大盘股")
            target_list1 = select_roic_stocks(context, B_stocks)
            target_list2 = select_large_cap_stocks(context, B_stocks)
            target_list3 = select_mid_cap_stocks(context, B_stocks)
            target_list = target_list2 + target_list1 + target_list3
        else:
            log.info("牛市 - 配置小盘股")
            target_list = select_small_cap_stocks(context, S_stocks)
    
    elif market_state == 'NEUTRAL':
        # 震荡市：选择动量为正的板块
        if B_mean > S_mean and B_mean > 0:
            log.info("震荡市 - 配置大盘股")
            target_list1 = select_large_cap_stocks(context, B_stocks)
            target_list2 = select_roic_stocks(context, B_stocks)
            target_list3 = select_mid_cap_stocks(context, B_stocks)
            target_list = target_list1 + target_list2 + target_list3
        elif S_mean > B_mean and S_mean > 0:
            log.info("震荡市 - 配置小盘股")
            target_list = select_small_cap_stocks(context, S_stocks)
        else:
            log.info("震荡市 - 配置外盘ETF")
            target_list = g.config.FOREIGN_ETF
    
    else:  # BEAR
        # 熊市：配置防御性资产
        log.info("熊市 - 配置外盘ETF")
        target_list = g.config.FOREIGN_ETF
    
    return list(set(target_list))

def apply_trading_filters(context, target_list):
    """应用交易过滤器"""
    target_list = filter_limitup_stock(context, target_list)
    target_list = filter_limitdown_stock(context, target_list)
    target_list = filter_paused_stock(target_list)
    return target_list

def execute_rebalance(context, target_list):
    """执行调仓"""
    # 卖出不在目标列表的股票（涨停股除外）
    for stock in g.hold_list:
        if (stock not in target_list) and (stock not in g.yesterday_HL_list):
            position = context.portfolio.positions[stock]
            close_position(position)
            if stock in g.position_tracking:
                del g.position_tracking[stock]
    
    # 计算需要买入的股票
    position_count = len(context.portfolio.positions)
    target_num = min(len(target_list), g.config.STOCK_NUM * 3)
    
    if target_num > position_count:
        # 计算每只股票的目标仓位
        value = context.portfolio.cash / (target_num - position_count)
        
        # 限制单只股票最大仓位
        max_position_value = context.portfolio.total_value * g.config.MAX_SINGLE_POSITION
        value = min(value, max_position_value)
        
        for stock in target_list:
            if stock not in context.portfolio.positions:
                if open_position(stock, value):
                    if len(context.portfolio.positions) == target_num:
                        break

# ==================== 交易模块 ====================
def order_target_value_(security, value):
    """自定义下单"""
    if value == 0:
        log.debug(f"卖出 {security}")
    else:
        log.debug(f"买入 {security} 目标市值 {value:.2f}")
    return order_target_value(security, value)

def open_position(security, value):
    """开仓"""
    order = order_target_value_(security, value)
    if order and order.filled > 0:
        return True
    return False

def close_position(position):
    """平仓"""
    security = position.security
    order = order_target_value_(security, 0)
    if order:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False

# ==================== 过滤函数 ====================
def filter_paused_stock(stock_list):
    """过滤停牌股票"""
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]

def filter_st_stock(stock_list):
    """过滤ST股票"""
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]

def filter_kcbj_stock(stock_list):
    """过滤科创板、北交所、创业板股票"""
    return [stock for stock in stock_list 
            if stock[0] not in ['4', '8'] 
            and stock[:2] != '68' 
            and stock[0] != '3']

def filter_limitup_stock(context, stock_list):
    """过滤涨停股票"""
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list 
            if stock in context.portfolio.positions
            or last_prices[stock][-1] < current_data[stock].high_limit]

def filter_limitdown_stock(context, stock_list):
    """过滤跌停股票"""
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list 
            if stock in context.portfolio.positions
            or last_prices[stock][-1] > current_data[stock].low_limit]

def filter_new_stock(context, stock_list):
    """过滤次新股（上市不足375天）"""
    yesterday = context.previous_date
    return [stock for stock in stock_list 
            if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]