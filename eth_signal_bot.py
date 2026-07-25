"""
结构破位+EMA趋势+RSI+动能K线策略 -- 实时信号侦测 + Discord通知（GitHub Actions版）

这个版本设计给 GitHub Actions 使用：
- DISCORD_WEBHOOK_URL 从环境变量读取（在GitHub仓库的Secrets里设置，不会写死在代码里）
- 状态文件(bot_state.json)存在仓库根目录，每次执行后由GitHub Actions自动提交回仓库
- 不需要自己的电脑或伺服器一直开着，完全由GitHub的伺服器排程执行

这个脚本每次执行会：
1. 从 OKX 抓最近的 1小时 + 5分钟 K棒
2. 用完全相同的策略逻辑（跟 Colab 回测同一套）算出所有历史信号
3. 检查最新一根「已收盘」的5分钟K棒是不是信号确认棒
4. 如果是新信号（之前没通知过），发 Discord 消息
5. 把已通知过的信号记录存到本地文件，避免重复通知（GitHub Actions会把这个文件提交回仓库保存）
"""

import ccxt
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
from datetime import datetime, timezone

# ============ 基本设置，请依你的状况修改 ============
SYMBOL = 'ETH/USDT'
EXCHANGE_ID = 'okx'

DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')

# 策略参数（跟 Colab 回测用的一致）
PIVOT_LEFT_RIGHT = 8
MOMENTUM_FACTOR = 1.0
RETEST_TOLERANCE = 0.005
MAX_WAIT_BARS = 288
STOP_LOOKBACK = 6
STOP_BUFFER = 0.0008
RR_RATIO = 2.0
EMA_WARMUP_BARS = 210

# 每次抓多少历史资料来重建当前结构状态（不用抓全部历史，抓够用就好）
FETCH_1H_DAYS = 180   # 抓最近180天的1小时线，找目前有效的结构位
FETCH_5M_DAYS = 45    # 抓最近45天的5分钟线，判断当前EMA/RSI/回踩状态

STATE_FILE = 'bot_state.json'
LOG_FILE = 'bot_log.txt'


def log(msg):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def send_discord(message):
    if not DISCORD_WEBHOOK_URL:
        log('⚠ 尚未设置 DISCORD_WEBHOOK_URL 环境变量，跳过发送，仅打印訊息：')
        log(message)
        return
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json={'content': message}, timeout=15)
        if resp.status_code not in (200, 204):
            log(f'⚠ Discord 发送失败: {resp.status_code} {resp.text}')
        else:
            log('✓ Discord 通知已发送')
    except Exception as e:
        log(f'⚠ Discord 发送时发生例外: {e}')


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'last_notified_confirm_time': None}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def fetch_ohlcv(symbol, timeframe, days, exchange_id=EXCHANGE_ID, limit=300):
    exchange = getattr(ccxt, exchange_id)()
    exchange.load_markets()
    now_ms = exchange.milliseconds()
    since = now_ms - days * 24 * 60 * 60 * 1000
    all_rows = []
    while since < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows += batch
        last_ts = batch[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        if len(batch) < limit:
            break
    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset='timestamp').set_index('timestamp').sort_index()
    return df


# ============ 策略逻辑（跟 Colab 回测同一套代码）============
def find_pivots(df, left=5, right=5):
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)
    for i in range(left, n - right):
        window_h = highs[i-left:i+right+1]
        if highs[i] == window_h.max():
            pivot_high[i] = highs[i]
        window_l = lows[i-left:i+right+1]
        if lows[i] == window_l.min():
            pivot_low[i] = lows[i]
    return pivot_high, pivot_low


def detect_structure_breaks(df, left=5, right=5):
    closes = df['close'].values
    n = len(df)
    pivot_high, pivot_low = find_pivots(df, left, right)

    current_support = None
    current_resistance = None
    support_broken = True
    resistance_broken = True

    events = []
    for i in range(n):
        confirm_i = i - right
        if confirm_i >= left:
            if not np.isnan(pivot_low[confirm_i]):
                current_support = pivot_low[confirm_i]
                support_broken = False
            if not np.isnan(pivot_high[confirm_i]):
                current_resistance = pivot_high[confirm_i]
                resistance_broken = False

        if current_support is not None and not support_broken and closes[i] < current_support:
            events.append({'time': df.index[i], 'type': 'bearish_break', 'level': current_support})
            support_broken = True

        if current_resistance is not None and not resistance_broken and closes[i] > current_resistance:
            events.append({'time': df.index[i], 'type': 'bullish_break', 'level': current_resistance})
            resistance_broken = True

    return events


def compute_indicators(df):
    df = df.copy()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    df['body'] = (df['close'] - df['open']).abs()
    df['avg_body20'] = df['body'].rolling(20).mean()
    return df


def check_short_conditions(row, momentum_factor):
    trend_ok = (row['close'] < row['ema21'] < row['ema50'] < row['ema200'])
    momentum_ok = (
        row['close'] < row['open'] and
        not np.isnan(row['avg_body20']) and
        row['body'] >= momentum_factor * row['avg_body20'] and
        row['close'] < row['ema21'] and row['close'] < row['ema50']
    )
    rsi_ok = row['rsi'] < 50
    return trend_ok and momentum_ok and rsi_ok


def check_long_conditions(row, momentum_factor):
    trend_ok = (row['close'] > row['ema21'] > row['ema50'] > row['ema200'])
    momentum_ok = (
        row['close'] > row['open'] and
        not np.isnan(row['avg_body20']) and
        row['body'] >= momentum_factor * row['avg_body20'] and
        row['close'] > row['ema21'] and row['close'] > row['ema50']
    )
    rsi_ok = row['rsi'] > 50
    return trend_ok and momentum_ok and rsi_ok


def find_entry_after_break(df5, break_time, break_level, direction, retest_tolerance, momentum_factor, max_wait_bars):
    window = df5.loc[df5.index >= break_time].iloc[:max_wait_bars]
    if len(window) < 5:
        return None
    in_retest_zone = False
    for t, row in window.iterrows():
        if direction == 'short':
            if row['high'] >= break_level * (1 - retest_tolerance):
                in_retest_zone = True
            if in_retest_zone and check_short_conditions(row, momentum_factor):
                return {'confirm_time': t, 'direction': 'short', 'confirm_close': row['close']}
        else:
            if row['low'] <= break_level * (1 + retest_tolerance):
                in_retest_zone = True
            if in_retest_zone and check_long_conditions(row, momentum_factor):
                return {'confirm_time': t, 'direction': 'long', 'confirm_close': row['close']}
    return None


def generate_signals(df5, break_events, retest_tolerance, momentum_factor, max_wait_bars, ema_warmup_bars):
    if len(df5) == 0:
        return []
    signals = []
    min_valid_time = df5.index.min() + pd.Timedelta(minutes=5 * ema_warmup_bars)
    for ev in break_events:
        if ev['time'] < min_valid_time:
            continue
        direction = 'short' if ev['type'] == 'bearish_break' else 'long'
        result = find_entry_after_break(df5, ev['time'], ev['level'], direction,
                                         retest_tolerance, momentum_factor, max_wait_bars)
        if result is not None:
            result['break_time'] = ev['time']
            result['break_level'] = ev['level']
            signals.append(result)
    return signals


def compute_stop_target(df5, signal, stop_lookback, stop_buffer, rr_ratio):
    idx_lookup = {t: i for i, t in enumerate(df5.index)}
    confirm_idx = idx_lookup.get(signal['confirm_time'])
    if confirm_idx is None or confirm_idx < stop_lookback:
        return None
    lookback = df5.iloc[confirm_idx - stop_lookback + 1: confirm_idx + 1]
    entry_ref_price = signal['confirm_close']  # 用确认K棒收盘价当参考（真正进场是下一根开盘，价格会很接近）

    if signal['direction'] == 'short':
        stop = lookback['high'].max() * (1 + stop_buffer)
        risk = stop - entry_ref_price
        target = entry_ref_price - rr_ratio * risk
    else:
        stop = lookback['low'].min() * (1 - stop_buffer)
        risk = entry_ref_price - stop
        target = entry_ref_price + rr_ratio * risk

    if risk <= 0:
        return None
    return {'entry_ref': entry_ref_price, 'stop': stop, 'target': target, 'risk_pct': risk / entry_ref_price * 100}


# ============ 主流程 ============
def main():
    log(f'=== 开始检查 {SYMBOL} 信号 ===')

    df_1h = fetch_ohlcv(SYMBOL, '1h', FETCH_1H_DAYS)
    df_5m = fetch_ohlcv(SYMBOL, '5m', FETCH_5M_DAYS)

    if len(df_1h) < 50 or len(df_5m) < 300:
        log('⚠ 抓到的历史资料太少，可能是网路或交易所问题，本次跳过')
        return

    # 去掉可能还没收盘的最后一根K棒，避免用到未完成的数据
    df_1h = df_1h.iloc[:-1]
    df_5m = df_5m.iloc[:-1]

    df_5m = compute_indicators(df_5m)

    break_events = detect_structure_breaks(df_1h, left=PIVOT_LEFT_RIGHT, right=PIVOT_LEFT_RIGHT)
    signals = generate_signals(df_5m, break_events, RETEST_TOLERANCE, MOMENTUM_FACTOR,
                                MAX_WAIT_BARS, EMA_WARMUP_BARS)

    if not signals:
        log('本次没有侦测到任何信号（历史范围内）')
        return

    latest_signal = max(signals, key=lambda s: s['confirm_time'])
    latest_bar_time = df_5m.index[-1]

    log(f'最新一根已收盘K棒时间: {latest_bar_time}')
    log(f'最新信号确认时间: {latest_signal["confirm_time"]}, 方向: {latest_signal["direction"]}')

    # 只有当最新信号「就是」最新一根已收盘K棒时，才算是全新的、值得通知的信号
    if latest_signal['confirm_time'] != latest_bar_time:
        log('最新信号不是最新K棒，代表目前没有新信号，本次不通知')
        return

    state = load_state()
    confirm_time_str = latest_signal['confirm_time'].isoformat()

    if state.get('last_notified_confirm_time') == confirm_time_str:
        log('这个信号已经通知过了，跳过重复通知')
        return

    calc = compute_stop_target(df_5m, latest_signal, STOP_LOOKBACK, STOP_BUFFER, RR_RATIO)
    if calc is None:
        log('⚠ 止损/止盈计算失败（可能是历史资料不足），本次不通知')
        return

    direction_cn = '做多 🟢' if latest_signal['direction'] == 'long' else '做空 🔴'
    my_time = latest_signal['confirm_time'] + pd.Timedelta(hours=8)

    message = (
        f'【{SYMBOL} 新信号】\n'
        f'方向: {direction_cn}\n'
        f'确认时间(MY): {my_time.strftime("%Y-%m-%d %H:%M")}\n'
        f'参考进场价: {calc["entry_ref"]:.2f}\n'
        f'止损参考: {calc["stop"]:.2f} (距离 {calc["risk_pct"]:.2f}%)\n'
        f'止盈参考: {calc["target"]:.2f} (风报比 1:{RR_RATIO})\n'
        f'\n⚠ 请以下一根5分钟K棒开盘价附近实际下单为准，此为策略参考价'
    )

    log('侦测到全新信号，准备发送通知')
    send_discord(message)

    state['last_notified_confirm_time'] = confirm_time_str
    save_state(state)
    log('=== 本次检查完成 ===\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f'❌ 脚本执行时发生错误: {e}')
        sys.exit(1)
