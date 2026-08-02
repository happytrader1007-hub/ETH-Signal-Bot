"""
结构破位+EMA趋势+RSI+动能K线策略 -- 实时信号侦测 + 交易追踪 + Discord通知（GitHub Actions版 v2）

这个版本比前一版多了「交易生命週期追踪」：
- 侦测到新信号时，记录成一笔「进行中(open)」的交易，存进 trades_data.json
- 每次执行都会检查所有「进行中」的交易，看后续价格有没有触及止损/止盈，
  一旦触及就更新状态成 target(止盈) / stop(止损) / timeout(超时平仓)
- trades_data.json 会包含完整交易清单 + 整体统计（总笔数、胜率、平均R等），
  设计给外部网站直接读取显示用

- DISCORD_WEBHOOK_URL 从环境变量读取（在GitHub仓库的Secrets里设置，不会写死在代码里）
- trades_data.json / bot_log.txt 存在仓库根目录，每次执行后由GitHub Actions自动提交回仓库
- 不需要自己的电脑或伺服器一直开着，完全由GitHub的伺服器排程执行
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
MAX_HOLD_BARS = 288       # 最长持仓24小时（跟回测一致），超过还没结果就算timeout
FEE_RATE = 0.0004

# 每次抓多少历史资料来重建当前结构状态（不用抓全部历史，抓够用就好）
FETCH_1H_DAYS = 180   # 抓最近180天的1小时线，找目前有效的结构位
FETCH_5M_DAYS = 45    # 抓最近45天的5分钟线，判断当前EMA/RSI/回踩状态，也用来结算open交易
RECENT_WINDOW_HOURS = 6  # 每次检查最近几小时内的新信号，避免因排程延迟而漏掉

TRADES_FILE = 'trades_data.json'
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


def load_trades_data():
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {'trades': []}
                data = json.loads(content)
                if 'trades' not in data:
                    data['trades'] = []
                return data
        except (json.JSONDecodeError, ValueError):
            log('⚠ trades_data.json 内容损坏或为空，重置为空白状态')
            return {'trades': []}
    return {'trades': []}


def save_trades_data(data):
    with open(TRADES_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


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


def shift_break_events(break_events, bar_minutes):
    """把破位事件的时间，从这根K棒的开盘时间，平移到收盘时间（也就是「真正知道破位发生」
    的那一刻），避免用还没走完的K棒资料去找回踩确认信号。1小时线传60。"""
    shifted = []
    for ev in break_events:
        new_ev = dict(ev)
        new_ev['time'] = ev['time'] + pd.Timedelta(minutes=bar_minutes)
        shifted.append(new_ev)
    return shifted


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
    entry_ref_price = signal['confirm_close']

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


# ============ 交易生命週期追踪 ============
def update_open_trades(trades, df5):
    """检查所有 open 状态的交易，看后续价格有没有触及止损/止盈/超时"""
    idx_lookup = {t: i for i, t in enumerate(df5.index)}
    high = df5['high'].values
    low = df5['low'].values
    close = df5['close'].values
    n = len(df5)

    for trade in trades:
        if trade['status'] != 'open':
            continue

        entry_time = pd.Timestamp(trade['entry_time'])
        entry_idx = idx_lookup.get(entry_time)
        if entry_idx is None:
            continue  # 进场时间不在目前抓到的资料范围内，跳过（下次资料更新后再看）

        direction = trade['direction']
        stop = trade['stop']
        target = trade['target']
        entry_price = trade['entry_price']
        risk = abs(entry_price - stop)

        end_idx = min(entry_idx + MAX_HOLD_BARS, n)
        resolved = False

        for j in range(entry_idx, end_idx):
            hit_stop = (low[j] <= stop) if direction == 'long' else (high[j] >= stop)
            hit_target = (high[j] >= target) if direction == 'long' else (low[j] <= target)

            if hit_stop or hit_target:
                exit_price = stop if hit_stop else target  # 同根K棒都触及时，保守假设止损先发生
                outcome = 'stop' if hit_stop else 'target'
                trade['status'] = outcome
                trade['exit_time'] = df5.index[j].isoformat()
                trade['exit_price'] = float(exit_price)
                gross_r = ((exit_price - entry_price) / risk if direction == 'long'
                           else (entry_price - exit_price) / risk)
                trade['r'] = round(gross_r - (FEE_RATE * 2 / (risk / entry_price)), 4)
                resolved = True
                break

        if not resolved and end_idx - entry_idx >= MAX_HOLD_BARS:
            # 超过最长持仓时间还没结果，视为 timeout，用最后一根收盘价结算
            last_idx = end_idx - 1
            trade['status'] = 'timeout'
            trade['exit_time'] = df5.index[last_idx].isoformat()
            trade['exit_price'] = float(close[last_idx])
            gross_r = ((close[last_idx] - entry_price) / risk if direction == 'long'
                       else (entry_price - close[last_idx]) / risk)
            trade['r'] = round(gross_r - (FEE_RATE * 2 / (risk / entry_price)), 4)


def compute_stats(trades):
    closed = [t for t in trades if t['status'] in ('target', 'stop', 'timeout')]
    open_trades = [t for t in trades if t['status'] == 'open']
    wins = [t for t in closed if t.get('r', 0) > 0]
    total_r = sum(t.get('r', 0) for t in closed)
    win_rate = len(wins) / len(closed) if closed else None
    avg_r = total_r / len(closed) if closed else None
    return {
        'total_trades': len(trades),
        'closed_trades': len(closed),
        'open_trades': len(open_trades),
        'wins': len(wins),
        'losses': len(closed) - len(wins),
        'win_rate': round(win_rate, 4) if win_rate is not None else None,
        'avg_r': round(avg_r, 4) if avg_r is not None else None,
        'total_r': round(total_r, 4),
    }


# ============ 主流程 ============
def main():
    log(f'=== 开始检查 {SYMBOL} 信号 ===')

    df_1h = fetch_ohlcv(SYMBOL, '1h', FETCH_1H_DAYS)
    df_5m = fetch_ohlcv(SYMBOL, '5m', FETCH_5M_DAYS)

    if len(df_1h) < 50 or len(df_5m) < 300:
        log('⚠ 抓到的历史资料太少，可能是网路或交易所问题，本次跳过')
        return

    # 去掉可能还没收盘的最后一根K棒
    df_1h = df_1h.iloc[:-1]
    df_5m_full = df_5m.iloc[:-1].copy()  # 保留完整版给「结算open交易」用（含最新价格）
    df_5m = compute_indicators(df_5m_full)

    latest_bar_time = df_5m.index[-1]
    current_price = float(df_5m['close'].iloc[-1])
    log(f'最新一根已收盘K棒时间: {latest_bar_time}, 现价: {current_price:.2f}')

    trades_data = load_trades_data()
    trades = trades_data.get('trades', [])
    known_entry_times = {t['entry_time'] for t in trades}

    # 1) 先结算所有 open 交易
    update_open_trades(trades, df_5m)

    # 2) 找新信号
    break_events_raw = detect_structure_breaks(df_1h, left=PIVOT_LEFT_RIGHT, right=PIVOT_LEFT_RIGHT)
    break_events = shift_break_events(break_events_raw, 60)  # 修正：把破位时间平移到1H真正收盘的那一刻
    signals = generate_signals(df_5m, break_events, RETEST_TOLERANCE, MOMENTUM_FACTOR,
                                MAX_WAIT_BARS, EMA_WARMUP_BARS)

    cutoff_time = latest_bar_time - pd.Timedelta(hours=RECENT_WINDOW_HOURS)
    recent_signals = sorted(
        [s for s in signals if cutoff_time <= s['confirm_time'] <= latest_bar_time],
        key=lambda s: s['confirm_time']
    )
    log(f'最近 {RECENT_WINDOW_HOURS} 小时内共有 {len(recent_signals)} 个信号')

    new_notifications = 0
    for sig in recent_signals:
        confirm_idx_lookup = {t: i for i, t in enumerate(df_5m.index)}
        confirm_idx = confirm_idx_lookup.get(sig['confirm_time'])
        if confirm_idx is None or confirm_idx + 1 >= len(df_5m):
            continue
        entry_idx = confirm_idx + 1
        entry_time = df_5m.index[entry_idx]
        entry_time_str = entry_time.isoformat()

        if entry_time_str in known_entry_times:
            continue  # 这笔已经记录过了

        calc = compute_stop_target(df_5m, sig, STOP_LOOKBACK, STOP_BUFFER, RR_RATIO)
        if calc is None:
            continue

        entry_price_actual = float(df_5m['open'].iloc[entry_idx])

        new_trade = {
            'entry_time': entry_time_str,
            'direction': sig['direction'],
            'entry_price': entry_price_actual,
            'stop': float(calc['stop']),
            'target': float(calc['target']),
            'status': 'open',
            'exit_time': None,
            'exit_price': None,
            'r': None,
        }
        trades.append(new_trade)
        known_entry_times.add(entry_time_str)

        direction_cn = '做多 🟢' if sig['direction'] == 'long' else '做空 🔴'
        my_time = entry_time + pd.Timedelta(hours=8)
        message = (
            f'【{SYMBOL} 新信号】\n'
            f'方向: {direction_cn}\n'
            f'进场时间(MY): {my_time.strftime("%Y-%m-%d %H:%M")}\n'
            f'进场价: {entry_price_actual:.2f}\n'
            f'止损: {calc["stop"]:.2f} (距离 {calc["risk_pct"]:.2f}%)\n'
            f'止盈: {calc["target"]:.2f} (风报比 1:{RR_RATIO})\n'
        )
        log(f'新交易 {entry_time_str}（{sig["direction"]}），准备发送通知')
        send_discord(message)
        new_notifications += 1

        # 这笔刚建立的交易也顺便检查一次有没有立刻被结算（针对补追的信号）
        update_open_trades([new_trade], df_5m)

    # 3) 汇总统计，存档
    trades.sort(key=lambda t: t['entry_time'])
    stats = compute_stats(trades)

    trades_data['trades'] = trades
    trades_data['stats'] = stats
    trades_data['current_price'] = current_price
    trades_data['last_updated'] = datetime.now(timezone.utc).isoformat()
    save_trades_data(trades_data)

    log(f'本次共发送 {new_notifications} 笔新通知；目前总交易数 {stats["total_trades"]}，'
        f'进行中 {stats["open_trades"]}，已结算 {stats["closed_trades"]}，胜率 {stats["win_rate"]}')
    log('=== 本次检查完成 ===\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f'❌ 脚本执行时发生错误: {e}')
        sys.exit(1)
