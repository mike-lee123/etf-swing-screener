import io
import json
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# 檢測環境中是否安裝 shioaji
try:
    import shioaji as sj
    SHIOAJI_AVAILABLE = True
except ImportError:
    SHIOAJI_AVAILABLE = False

# ==============================================================================
# 0. 系統配置與冷卻控制器 (Rate Limiter)
# ==============================================================================
st.set_page_config(
    page_title="ETF 成分股波段選股儀表板",
    page_icon="📈",
    layout="wide"
)

def clean_ascii(text: str) -> str:
    """自動過濾字串多餘空白與非 ASCII 符號"""
    if not text:
        return ""
    return re.sub(r"[^\x00-\x7F]+", "", str(text)).strip()

class RequestCooldownManager:
    """冷卻節流器：防止連續高頻調用導致 Yahoo 或券商伺服器阻斷"""
    def __init__(self, min_interval_seconds=0.3):
        self.min_interval = min_interval_seconds
        self.last_request_time = 0.0

    def wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

cooldown_manager = RequestCooldownManager(min_interval_seconds=0.3)

# ==============================================================================
# 1. 6 檔 ETF 完整成分股宇宙 (自動抓取最新持股)、分層標籤與中文名稱
# ==============================================================================
# 第一層：核心權值主升浪 / 第二層：法人共識主升股 (主動式 ETF 重疊) / 第三層：高彈性突破黑馬 (00733)
ETF_LIST = ["0050", "0052", "00935", "00981A", "00982A", "00733"]
HOLDINGS_CACHE_FILE = Path(__file__).with_name("etf_holdings_cache.json")
MONEYDJ_HOLDINGS_URL = "https://www.moneydj.com/ETF/X/Basic/Basic0007B.xdjhtm?etfid={etf}.TW"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

# 線上抓取與本地快取皆失效時的最後備援 (各 ETF 前幾大權重股)
FALLBACK_HOLDINGS = {
    "0050": {"2330": "台積電", "2454": "聯發科", "2317": "鴻海", "2308": "台達電", "2382": "廣達", "2881": "富邦金", "2882": "國泰金"},
    "0052": {"2330": "台積電", "2454": "聯發科", "2317": "鴻海", "2382": "廣達", "3034": "聯詠", "2379": "瑞昱", "3008": "大立光"},
    "00935": {"2330": "台積電", "2454": "聯發科", "2308": "台達電", "6669": "緯穎", "2382": "廣達", "3653": "健策", "3231": "緯創"},
    "00981A": {"2330": "台積電", "2454": "聯發科", "3653": "健策", "6669": "緯穎", "3035": "智原", "3443": "創意", "3324": "雙鴻"},
    "00982A": {"2330": "台積電", "2317": "鴻海", "3653": "健策", "3035": "智原", "8299": "群聯", "3443": "創意", "6515": "穎崴"},
    "00733": {"3653": "健策", "3324": "雙鴻", "6515": "穎崴", "8299": "群聯", "3035": "智原", "3529": "力旺", "2467": "志聖"},
}

def fetch_etf_holdings_moneydj(etf: str):
    """從 MoneyDJ 全部持股頁抓取單一 ETF 最新成分股，回傳 ({代碼: 名稱}, 資料日期)"""
    resp = requests.get(MONEYDJ_HOLDINGS_URL.format(etf=etf), headers=HTTP_HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    tables = pd.read_html(io.StringIO(resp.text))
    table = next(t for t in tables if "個股名稱" in t.columns)
    # 例如 "台積電(2330.TW)"；期貨等非個股列不符合格式會被略過
    parsed = table["個股名稱"].astype(str).str.extract(r"^(.+)\((\d{4,6}[A-Z]?)\.TWO?\)$").dropna()
    holdings = {code: name.strip() for name, code in zip(parsed[0], parsed[1])}
    if not holdings:
        raise ValueError("持股表格解析結果為空")
    date_match = re.search(r"資料日期[^\d]*([\d/]+)", resp.text)
    return holdings, (date_match.group(1) if date_match else "未知")

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_etf_holdings():
    """逐檔抓取最新持股；失敗時依序退回本地快取檔、內建備援名單"""
    try:
        disk_cache = json.loads(HOLDINGS_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        disk_cache = {}

    etf_holdings, meta = {}, {}
    for etf in ETF_LIST:
        try:
            holdings, data_date = fetch_etf_holdings_moneydj(etf)
            etf_holdings[etf] = holdings
            meta[etf] = {"source": "MoneyDJ 即時", "date": data_date}
            disk_cache[etf] = {"holdings": holdings, "date": data_date}
        except Exception as e:
            print(f"[ETF] {etf} 持股抓取失敗: {type(e).__name__}: {e}", flush=True)
            if etf in disk_cache:
                etf_holdings[etf] = disk_cache[etf]["holdings"]
                meta[etf] = {"source": "本地快取", "date": disk_cache[etf]["date"]}
            else:
                etf_holdings[etf] = FALLBACK_HOLDINGS[etf]
                meta[etf] = {"source": "內建備援(前幾大)", "date": "-"}

    try:
        HOLDINGS_CACHE_FILE.write_text(json.dumps(disk_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return etf_holdings, meta

def get_etf_universe():
    holdings_map, _ = load_etf_holdings()
    etf_holdings = {etf: list(h.keys()) for etf, h in holdings_map.items()}
    stock_names = {code: name for h in holdings_map.values() for code, name in h.items()}

    core_pool = set(etf_holdings["0050"] + etf_holdings["0052"] + etf_holdings["00935"])
    active_overlap = set(etf_holdings["00981A"]).intersection(set(etf_holdings["00982A"]))
    momentum_pool = set(etf_holdings["00733"])

    all_symbols = set()
    for stocks in etf_holdings.values():
        all_symbols.update(stocks)

    stock_tags = {}
    for sym in all_symbols:
        tags = []
        if sym in core_pool:
            tags.append("核心權值")
        if sym in active_overlap:
            tags.append("法人共識")
        if sym in momentum_pool and sym not in etf_holdings["0050"]:
            tags.append("突破黑馬")
        stock_tags[sym] = tags if tags else ["一般持股"]

    stock_etfs = {sym: [etf for etf in ETF_LIST if sym in etf_holdings[etf]] for sym in all_symbols}
    return sorted(list(all_symbols)), stock_tags, stock_names, stock_etfs

# ==============================================================================
# 2. 永豐 Shioaji 通道連線 (採用併武相同連線架構)
# ==============================================================================
def connect_shioaji(api_key: str, secret_key: str, is_simulation: bool = True):
    """採用併武穩定架構登入 Shioaji"""
    if not SHIOAJI_AVAILABLE:
        return None, "❌ 尚未安裝 shioaji 套件 (請執行: pip install shioaji)"

    clean_key = clean_ascii(api_key)
    clean_sec = clean_ascii(secret_key)

    if not clean_key or not clean_sec:
        return None, "⚠️ 請先填寫 API Key 與 Secret Key"

    try:
        api = sj.Shioaji(simulation=is_simulation)
        accounts = api.login(clean_key, clean_sec)

        if not accounts:
            return None, "❌ 登入失敗：未取得交易帳號，請確認金鑰權限。"

        # shioaji 1.x 的 login() 會在背景自動下載合約，不可再呼叫 fetch_contracts()
        # (否則會與背景下載互搶而拋出 "exclusive access lost")，改為輪詢等待合約就緒
        deadline = time.time() + 60
        while True:
            try:
                if api.Contracts.Stocks.get("2330") is not None:
                    break
            except Exception:
                pass
            if time.time() > deadline:
                api.logout()
                return None, "❌ 合約下載逾時 (60 秒)，請稍後再試"
            time.sleep(0.5)

        mode_name = "模擬模式" if is_simulation else "正式模式"
        return api, f"🟢 Shioaji 實戰通道已連線 ({mode_name})"
    except Exception as e:
        print(f"[SJ] error: {type(e).__name__}: {e}", flush=True)
        return None, f"❌ 永豐通道連線失敗: {type(e).__name__}: {e}"

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']

def fetch_shioaji_daily(symbol: str, sj_api, lookback_days=180) -> pd.DataFrame:
    """以 Shioaji 抓取個股 K 線並合併為日 K；資料不足 70 天時回傳空表交由 Yahoo 備援"""
    cooldown_manager.wait()
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        contract = sj_api.Contracts.Stocks.get(symbol)
        if contract:
            kbars = sj_api.kbars(contract=contract, start=start_date)
            df = pd.DataFrame({**kbars})
            if not df.empty:
                # kbars 回傳的是 1 分 K，需合併為日 K 才能套用日線指標
                df['ts'] = pd.to_datetime(df['ts'])
                df = df.set_index('ts').resample('1D').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min',
                    'Close': 'last', 'Volume': 'sum'
                }).dropna()
                df.index.name = 'Date'
                if len(df) >= 70:
                    return df
    except Exception as e:
        print(f"[SJ] {symbol} kbars error: {type(e).__name__}: {e}", flush=True)
    return pd.DataFrame()

def download_yahoo_batch(symbols) -> dict:
    """Yahoo Finance 批次下載日 K；先試上市 .TW，抓不到的再試上櫃 .TWO"""
    result = {}
    pending = list(symbols)
    for suffix in (".TW", ".TWO"):
        if not pending:
            break
        yf_tickers = [f"{s}{suffix}" for s in pending]
        try:
            data = yf.download(yf_tickers, period="8mo", interval="1d", progress=False,
                               group_by="ticker", threads=True)
        except Exception as e:
            print(f"[YF] batch {suffix} error: {type(e).__name__}: {e}", flush=True)
            continue
        for sym, ticker in zip(pending, yf_tickers):
            try:
                df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
                df = df[OHLCV].dropna()
                if not df.empty:
                    result[sym] = df
            except KeyError:
                continue
        pending = [s for s in pending if s not in result]
    return result

# ==============================================================================
# 3. 量價與指標篩選管線 (帶 15 分鐘 Streamlit 快取保護)
# ==============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def run_screening_pipeline(_sj_api, data_source: str, min_vol_ratio: float, rs_window: int = 20):
    # data_source 參與快取鍵 (_sj_api 不參與)，讓 Yahoo 與永豐的結果分開快取
    tickers, stock_tags, stock_names, stock_etfs = get_etf_universe()

    # 下載大盤基準 (^TWII) 計算相對強弱超額
    cooldown_manager.wait()
    bench_df = yf.download("^TWII", period="8mo", interval="1d", progress=False)
    if isinstance(bench_df.columns, pd.MultiIndex):
        bench_df.columns = bench_df.columns.get_level_values(0)
    bench_close = bench_df['Close'].dropna()
    bench_rs = (bench_close.iloc[-1] / bench_close.iloc[-rs_window] - 1) if len(bench_close) >= rs_window else 0

    results = []
    progress_bar = st.progress(0)

    # 先試 Shioaji，取得不足的標的再一次批次交由 Yahoo 補齊
    price_data = {}
    if _sj_api is not None:
        for idx, sym in enumerate(tickers):
            progress_bar.progress((idx + 1) / len(tickers), text=f"永豐 K 線下載中 {sym} ({idx + 1}/{len(tickers)})")
            df = fetch_shioaji_daily(sym, _sj_api)
            if not df.empty:
                price_data[sym] = df
        print(f"[SJ] {len(price_data)}/{len(tickers)} 檔取得永豐日K，其餘改用 Yahoo", flush=True)
    missing = [s for s in tickers if s not in price_data]
    if missing:
        progress_bar.progress(1.0, text=f"Yahoo Finance 批次下載 {len(missing)} 檔...")
        price_data.update(download_yahoo_batch(missing))

    for idx, sym in enumerate(tickers):
        df = price_data.get(sym, pd.DataFrame())

        if df.empty or len(df) < 70:
            continue

        close = df['Close']
        volume = df['Volume']
        low = df['Low']

        # 計算均線
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        vol_ma5 = volume.rolling(5).mean()

        curr_close = float(close.iloc[-1])
        curr_vol = float(volume.iloc[-1])
        prev_vol_ma5 = float(vol_ma5.iloc[-2]) if len(vol_ma5) >= 2 else 1.0

        # --- 1. 趨勢基本過濾 (收盤站穩月線，季線走平或翻揚) ---
        ma60_slope_up = ma60.iloc[-1] >= (ma60.iloc[-5] * 0.998)
        cond_trend = (curr_close >= ma20.iloc[-1]) and (curr_close >= ma60.iloc[-1]) and ma60_slope_up

        if not cond_trend:
            continue

        # --- 2. 相對強弱指標 (RS) ---
        stock_rs = (curr_close / close.iloc[-rs_window] - 1)
        rs_excess = stock_rs - bench_rs
        if rs_excess < -0.02:  # 容許落後大盤在 2% 以內
            continue

        # --- 3. 布林通道帶寬收斂評估 ---
        std20 = close.rolling(20).std()
        bandwidth = ((ma20 + std20 * 2) - (ma20 - std20 * 2)) / ma20
        min_bw_60 = bandwidth.iloc[-60:].min()
        is_bw_compressed = bandwidth.iloc[-1] <= (min_bw_60 * 1.30)
        is_bw_turning = bandwidth.iloc[-1] > bandwidth.iloc[-2]

        # --- 4. 動態量能判定 (核心權值門檻較為平緩) ---
        tags_str = " / ".join(stock_tags.get(sym, ["一般"]))
        target_vol_ratio = 1.2 if "核心權值" in tags_str else min_vol_ratio
        cond_vol = curr_vol >= (prev_vol_ma5 * target_vol_ratio)

        # --- 5. 訊號階梯分級 ---
        status = None
        triggers = []
        vol_multiple = curr_vol / prev_vol_ma5 if prev_vol_ma5 > 0 else 1.0

        if cond_vol and is_bw_turning:
            status = "🔥 強勢爆發"
            triggers.append(f"爆量({round(vol_multiple, 1)}x)+布林張口")
        elif cond_vol:
            status = "⚡ 帶量突破"
            triggers.append(f"量增突破({round(vol_multiple, 1)}x)")
        elif is_bw_compressed:
            status = "⏳ 壓縮蓄勢"
            triggers.append("布林極致壓縮(待量)")
        elif rs_excess > 0:
            status = "📈 趨勢偏多"
            triggers.append("均線多頭+RS強")

        if not status:
            continue

        # 關鍵支撐與停損點位
        recent_swing_low = float(low.iloc[-10:].min())
        key_support = max(recent_swing_low, float(ma20.iloc[-1]))
        suggested_stop_loss = min(key_support * 0.985, curr_close * 0.94)

        # 股票中文名稱
        stock_name = stock_names.get(sym, sym)

        results.append({
            "代碼": sym,
            "名稱": stock_name,
            "狀態": status,
            "所屬層級": tags_str,
            "持有ETF": " / ".join(stock_etfs.get(sym, [])),
            "收盤價": round(curr_close, 2),
            "20日超額RS(%)": round(float(rs_excess) * 100, 2),
            "成交量比率": round(vol_multiple, 2),
            "發動特徵": " + ".join(triggers),
            "關鍵支撐": round(key_support, 2),
            "建議停損點": round(suggested_stop_loss, 2)
        })

    progress_bar.empty()
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        status_weight = {"🔥 強勢爆發": 4, "⚡ 帶量突破": 3, "⏳ 壓縮蓄勢": 2, "📈 趨勢偏多": 1}
        res_df['rank_score'] = res_df['狀態'].map(status_weight) * 1000 + res_df['20日超額RS(%)']
        res_df = res_df.sort_values(by="rank_score", ascending=False).drop(columns=['rank_score']).reset_index(drop=True)

    return res_df

# ==============================================================================
# 4. Streamlit 介面主體與狀態控制
# ==============================================================================
st.title("📈 ETF 成分股多層波段選股儀表板")
st.caption("涵蓋核心權值 (0050/0052/00935)、法人共識 (00981A/00982A) 與高彈性黑馬 (00733)")

with st.spinner("抓取各 ETF 最新成分股中..."):
    _holdings_map, _holdings_meta = load_etf_holdings()
with st.expander(f"📦 成分股宇宙：共 {len({c for h in _holdings_map.values() for c in h})} 檔（去重後）"):
    st.dataframe(
        pd.DataFrame([
            {"ETF": etf, "成分股數": len(_holdings_map[etf]),
             "資料日期": _holdings_meta[etf]["date"], "來源": _holdings_meta[etf]["source"]}
            for etf in ETF_LIST
        ]),
        hide_index=True
    )

# 初始化 session 狀態
if "sj_instance" not in st.session_state:
    st.session_state["sj_instance"] = None
if "sj_status_msg" not in st.session_state:
    st.session_state["sj_status_msg"] = "ℹ️ 尚未連線至永豐金，預設運行於 Yahoo Finance 備援模式"

# 側邊欄配置
with st.sidebar:
    st.header("⚙️ 引擎與參數設定")

    with st.expander("永豐 Shioaji API 設定", expanded=True):
        sj_api_key = st.text_input("API Key", type="password")
        sj_secret_key = st.text_input("Secret Key", type="password")
        # 預設為 True，完全對齊併武的連線設定
        is_sim_mode = st.checkbox("使用模擬環境 (Simulation)", value=True, help="對齊併武系統模擬實戰通道")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            btn_connect = st.button("🔗 連線通道", width="stretch")
        with col_btn2:
            btn_disconnect = st.button("🔌 中斷", width="stretch")

    # 處理連線邏輯
    if btn_connect:
        with st.spinner("連線永豐金伺服器中..."):
            api_obj, msg = connect_shioaji(sj_api_key, sj_secret_key, is_simulation=is_sim_mode)
            st.session_state["sj_instance"] = api_obj
            st.session_state["sj_status_msg"] = msg

    if btn_disconnect:
        if st.session_state["sj_instance"] is not None:
            try:
                st.session_state["sj_instance"].logout()
            except Exception:
                pass
        st.session_state["sj_instance"] = None
        st.session_state["sj_status_msg"] = "ℹ️ 已切斷永豐連線，目前運行於 Yahoo Finance 備援模式"

    # 即時顯示 API 狀態
    st.info(st.session_state["sj_status_msg"])

    vol_multiplier = st.slider("中小型突破量能倍數 (vs 5MA量)", 1.1, 2.5, 1.3, step=0.1)

    selected_status = st.multiselect(
        "欲顯示的發動狀態",
        options=["🔥 強勢爆發", "⚡ 帶量突破", "⏳ 壓縮蓄勢", "📈 趨勢偏多"],
        default=["🔥 強勢爆發", "⚡ 帶量突破", "⏳ 壓縮蓄勢", "📈 趨勢偏多"]
    )

    top_candidates_count = st.slider("頂部精選展示檔數", 3, 8, 4)

    if st.button("🔄 清除快取並重新掃描", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# 執行管線
with st.spinner("盤後量價與指標過濾管線執行中，請稍候..."):
    sj_instance = st.session_state["sj_instance"]
    data_source = "shioaji" if sj_instance is not None else "yahoo"
    screened_df = run_screening_pipeline(sj_instance, data_source, min_vol_ratio=vol_multiplier)

# 呈現結果
if not screened_df.empty:
    filtered_df = screened_df[screened_df['狀態'].isin(selected_status)].reset_index(drop=True)

    if not filtered_df.empty:
        top_df = filtered_df.head(top_candidates_count)

        st.subheader(f"🎯 重點發動與蓄勢精選 (Top {len(top_df)})")
        cols = st.columns(len(top_df))
        for i, row in top_df.iterrows():
            with cols[i]:
                st.metric(
                    label=f"{row['代碼']} {row['名稱']} | {row['狀態']}",
                    value=f"{row['收盤價']} 元",
                    delta=f"RS +{row['20日超額RS(%)']}%"
                )
                st.caption(f"🛡️ 停損: {row['建議停損點']} | 支撐: {row['關鍵支撐']}")

        st.markdown("---")
        st.subheader("📋 完整量化波段清單與訊號明細")
        st.dataframe(
            filtered_df,
            column_config={
                "代碼": st.column_config.TextColumn("代碼", width="small"),
                "名稱": st.column_config.TextColumn("名稱", width="small"),
                "成交量比率": st.column_config.ProgressColumn(
                    "成交量比率 (倍數)",
                    format="%.2f x",
                    min_value=0.5,
                    max_value=3.5
                ),
                "20日超額RS(%)": st.column_config.NumberColumn(
                    "20日相對大盤超額",
                    format="%.2f %%"
                )
            },
            width="stretch",
            hide_index=True
        )
    else:
        st.warning("篩選結果中沒有符合您所選勾選狀態（" + "、".join(selected_status) + "）的標的。請勾選更多狀態或調降量能倍數。")
else:
    st.warning("目前市場無標的符合基礎趨勢條件。")