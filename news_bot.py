"""
Kalshi News Event Bot
Monitors RSS feeds for breaking news, uses Claude Haiku to identify
market-repricing events, and executes limit orders on Kalshi within seconds.

Two-tier architecture:
  Tier 1 (<50ms): Regex keyword map → immediate order on known market
  Tier 2 (~400ms): Claude Haiku classification → order for ambiguous headlines

Feeds: BBC, Bloomberg, Politico, NPR (ETag-based conditional GETs)
"""

import asyncio
import base64
import json
import logging
import os
import time
from flask import Flask, jsonify
import threading
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import anthropic
import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv
from risk_guard import RiskManager

load_dotenv()

# ── Quant Fund Shadow Evaluators ─────────────────────────────────────────
try:
    from bayesian_updater import BayesianUpdater
    from ensemble_model import EnsembleModel
    from time_decay_edge import calculate_time_weighted_edge
    from correlation_matrix import CorrelationTracker
    from vpin_toxicity import VPINTracker
    from market_impact import estimate_market_impact
    from feature_engine import FeatureEngine
    from portfolio_optimizer import PortfolioOptimizer
    _quant_modules_available = True
    _bayesian = BayesianUpdater()
    _ensemble = EnsembleModel()
    _correlation = CorrelationTracker()
    _vpin = VPINTracker()
    _features = FeatureEngine()
    _portfolio = PortfolioOptimizer()
except ImportError:
    _quant_modules_available = False


# ── Shadow Logging ────────────────────────────────────────────────────────────
SHADOW_LOG_FILE = os.getenv("SHADOW_LOG_FILE", "shadow_log.jsonl")

def shadow_log(opportunity: dict, taken: bool, reason: str = ""):
    entry = {"ts": time.time(), "taken": taken, "reason": reason, **opportunity}
    try:
        with open(SHADOW_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass


# ── Virtual Portfolio Testing ─────────────────────────────────────────────
VIRTUAL_PORTFOLIO_FILE = os.getenv("VIRTUAL_PORTFOLIO_FILE", "virtual_portfolios.jsonl")

VIRTUAL_PORTFOLIOS = [
    {"name": "aggressive", "kelly": 1.0, "min_edge": 0.02, "early_exit": 0.99},
    {"name": "moderate", "kelly": 0.5, "min_edge": 0.05, "early_exit": 0.93},
    {"name": "conservative", "kelly": 0.25, "min_edge": 0.08, "early_exit": 0.90},
    {"name": "original_v1", "kelly": 1.0, "min_edge": 0.03, "early_exit": 0.99},
    {"name": "high_edge", "kelly": 0.5, "min_edge": 0.10, "early_exit": 0.93},
    {"name": "ultra_conservative", "kelly": 0.25, "min_edge": 0.12, "early_exit": 0.90},
]

def evaluate_virtual_portfolios(opportunity: dict):
    """Evaluate what each virtual portfolio would do with this opportunity."""
    import json, time as _time
    edge = opportunity.get("edge", 0)
    price = opportunity.get("price", 0)
    results = []
    for vp in VIRTUAL_PORTFOLIOS:
        would_trade = edge >= vp["min_edge"]
        would_exit_early = price >= vp["early_exit"] * 100
        results.append({
            "portfolio": vp["name"],
            "would_trade": would_trade,
            "would_exit_early": would_exit_early,
            "kelly": vp["kelly"],
            "min_edge": vp["min_edge"],
        })
    entry = {
        "ts": _time.time(),
        "opportunity": opportunity,
        "portfolios": results,
    }
    try:
        with open(VIRTUAL_PORTFOLIO_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass

# ─── Regime Detection — pause trading during extreme volatility ────────────
import statistics as _stats

REGIME_WINDOW = int(os.getenv("REGIME_WINDOW", "20"))
REGIME_THRESHOLD = float(os.getenv("REGIME_THRESHOLD", "3.0"))
_regime_prices: list[float] = []

def check_regime(price: float) -> str:
    """Returns 'CALM', 'ELEVATED', or 'CRASH'. Skip trades during CRASH."""
    _regime_prices.append(price)
    if len(_regime_prices) > REGIME_WINDOW:
        _regime_prices.pop(0)
    if len(_regime_prices) < 5:
        return "CALM"
    rets = [(b - a) / a for a, b in zip(_regime_prices[:-1], _regime_prices[1:])]
    if not rets:
        return "CALM"
    mu = _stats.mean(rets)
    sd = _stats.stdev(rets) if len(rets) > 1 else 0.01
    z = abs(rets[-1] - mu) / max(sd, 0.0001)
    if z > REGIME_THRESHOLD:
        return "CRASH"
    elif z > REGIME_THRESHOLD * 0.6:
        return "ELEVATED"
    return "CALM"



# ── Early Exit Logic ─────────────────────────────────────────────────────────
EARLY_EXIT_THRESHOLD = float(os.getenv("EARLY_EXIT_THRESHOLD", "0.93"))

def should_early_exit(current_price_cents: float) -> bool:
    """Exit position early at 93c+ to lock in profit instead of holding to settlement."""
    return current_price_cents >= EARLY_EXIT_THRESHOLD * 100

# ── Circuit Breakers ─────────────────────────────────────────────────────────
CONSECUTIVE_LOSS_PAUSE = int(os.getenv("CONSECUTIVE_LOSS_PAUSE", "3"))
DAILY_DRAWDOWN_PAUSE_PCT = float(os.getenv("DAILY_DRAWDOWN_PAUSE_PCT", "0.05"))

_consecutive_losses = 0
_daily_pnl = 0.0
_circuit_paused_until = 0

def check_circuit_breaker() -> bool:
    """Returns True if trading should be paused."""
    import time as _time
    global _consecutive_losses, _daily_pnl, _circuit_paused_until
    if _time.time() < _circuit_paused_until:
        return True
    if _consecutive_losses >= CONSECUTIVE_LOSS_PAUSE:
        return True
    # Use PAPER_BALANCE if available, else 5000
    _balance = globals().get("PAPER_BALANCE", 2000)
    if _daily_pnl < -DAILY_DRAWDOWN_PAUSE_PCT * _balance:
        return True
    return False

def record_trade_result(won: bool, pnl: float):
    """Update circuit breaker state after each trade result."""
    global _consecutive_losses, _daily_pnl
    _daily_pnl += pnl
    if won:
        _consecutive_losses = 0
    else:
        _consecutive_losses += 1
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Risk Guard ────────────────────────────────────────────────────────────────
risk_manager = RiskManager()

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class Config:
    KALSHI_API_KEY_ID: str = os.environ["KALSHI_API_KEY_ID"]
    KALSHI_PRIVATE_KEY_PEM: str = os.environ["KALSHI_PRIVATE_KEY_PEM"]
    ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]

    PAPER_MODE: bool = os.environ.get("PAPER_MODE", "true").lower() == "true"
    PAPER_BALANCE: float = float(os.environ.get("PAPER_STARTING_BALANCE", "2000.0"))
    MAX_TRADE_USD: float = float(os.environ.get("MAX_TRADE_USD", "50.0"))
    MIN_EDGE_PCT: float = float(os.environ.get("MIN_EDGE_PCT", "0.05"))  # min % edge vs current price
    MAKER_FEE: float = float(os.environ.get("MAKER_FEE", "0.0175"))
    KELLY_FRACTION: float = float(os.environ.get("KELLY_FRACTION", "0.25"))


# ─── News Feeds ───────────────────────────────────────────────────────────────

FEEDS = [
    {"url": "https://feeds.bbci.co.uk/news/rss.xml",         "name": "BBC Top",      "poll_sec": 5},
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml",   "name": "BBC World",    "poll_sec": 10},
    {"url": "https://feeds.npr.org/1001/rss.xml",             "name": "NPR News",     "poll_sec": 10},
    {"url": "https://feeds.npr.org/1014/rss.xml",             "name": "NPR Politics", "poll_sec": 15},
    {"url": "https://feeds.apnews.com/rss/APNewsTopHeadlines", "name": "AP News",      "poll_sec": 60},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "name": "NYT Business","poll_sec": 60},
    {"url": "https://www.politico.com/rss/politicopicks.xml", "name": "Politico",     "poll_sec": 30},
]


# ─── Tier 1 Keyword Map ───────────────────────────────────────────────────────
# Pattern → (series_ticker, side_to_buy)
# side "yes" = buy YES (event happens), "no" = buy NO (event does not happen)

KEYWORD_MAP = [
    (re.compile(r"fed.{0,40}(raises?|hikes?|increase).{0,20}rate",  re.I), "KXFED", "yes"),
    (re.compile(r"fed.{0,40}(cuts?|lowers?|reduces?|decrease).{0,20}rate", re.I), "KXFED", "no"),
    (re.compile(r"cpi.{0,30}(higher|hot|above|exceed|surpass|jump)",  re.I), "KXCPI", "yes"),
    (re.compile(r"cpi.{0,30}(lower|cool|below|miss|fall|drop)",       re.I), "KXCPI", "no"),
    (re.compile(r"bitcoin.{0,20}\b(9[0-9],|[1-9]\d{2},)",            re.I), "KXBTC", "yes"),
    (re.compile(r"bitcoin.{0,30}(crash|plunge|dump|collapse)",        re.I), "KXBTC", "no"),
    (re.compile(r"(recession|gdp.{0,20}contract|gdp.{0,20}negative)", re.I), "KXGDP", "no"),
    (re.compile(r"(gdp.{0,20}(beat|above|exceed|surpass|strong))",   re.I), "KXGDP", "yes"),
    (re.compile(r"(iran.{0,30}(attack|strike|war|bomb|missile))",     re.I), "KXIRAN", "yes"),
    (re.compile(r"ceasefire.{0,30}(iran|israel|hamas|russia|ukraine)",re.I), "KXIRAN", "no"),
]


# ─── Market Catalog for Claude ────────────────────────────────────────────────

MARKET_CATALOG = """
KXBTC: Bitcoin daily/hourly price ranges (e.g., "Will BTC be above $90,000 at 5pm EDT?")
KXETH: Ethereum daily price ranges at 5pm EDT
KXFED: Federal funds rate at next FOMC meeting (will the Fed raise/cut/hold?)
KXCPI: Consumer Price Index monthly release (will CPI rise more/less than X%?)
KXGDP: Real GDP quarterly growth rate
KXIRAN: Iran-related geopolitical events (nuclear deal, military strikes)
KXNBA: NBA game outcomes (who wins, point spreads)
KXNFL: NFL game outcomes
KXMLB: MLB game outcomes
KXNHL: NHL game outcomes
KXTORNADO: Monthly US tornado counts
KXHIGHNY: Daily high temperature in New York City
KXHIGHCHI: Daily high temperature in Chicago
KXHIGHMIA: Daily high temperature in Miami
KXHIGHLAX: Daily high temperature in Los Angeles
KXHIGHDEN: Daily high temperature in Denver
"""


# ─── Kalshi Auth ──────────────────────────────────────────────────────────────

def load_private_key(pem: str):
    return serialization.load_pem_private_key(pem.replace("\\n", "\n").encode(), password=None)


def _sign(private_key, method: str, path: str) -> tuple[str, str]:
    ts = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    msg = (ts + method + path).encode()
    sig = private_key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())
    return ts, base64.b64encode(sig).decode()


def _headers(private_key, key_id: str, method: str, path: str) -> dict:
    ts, sig = _sign(private_key, method, path)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }


# ─── Kalshi API ───────────────────────────────────────────────────────────────

async def get_series_markets(http: httpx.AsyncClient, private_key, key_id: str, series: str) -> list[dict]:
    path = "/trade-api/v2/markets"
    r = await http.get(BASE_URL + "/markets",
        params={"series_ticker": series, "status": "open", "limit": 50},
        headers=_headers(private_key, key_id, "GET", path))
    r.raise_for_status()
    return r.json().get("markets", [])


async def get_orderbook(http: httpx.AsyncClient, private_key, key_id: str, ticker: str) -> Optional[dict]:
    path = f"/trade-api/v2/markets/{ticker}/orderbook"
    try:
        r = await http.get(BASE_URL + f"/markets/{ticker}/orderbook",
            params={"depth": 3},
            headers=_headers(private_key, key_id, "GET", path))
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Orderbook failed {ticker}: {e}")
        return None


async def place_order(http: httpx.AsyncClient, private_key, key_id: str,
                      ticker: str, side: str, price_cents: int, count: int,
                      paper_mode: bool) -> Optional[dict]:
    if paper_mode:
        log.info(f"[PAPER ORDER] {ticker} {side} @ {price_cents}¢ x{count}")
        return {"order_id": f"paper-{uuid.uuid4()}", "status": "paper"}

    path = "/trade-api/v2/portfolio/orders"
    body = {
        "ticker": ticker,
        "side": side,
        "action": "buy",
        "count": count,
        "yes_price" if side == "yes" else "no_price": price_cents,
        "time_in_force": "fill_or_kill",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": str(uuid.uuid4()),
    }
    try:
        r = await http.post(BASE_URL + "/portfolio/orders", json=body,
            headers=_headers(private_key, key_id, "POST", path))
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Order failed {ticker}: {e}")
        return None


# ─── Market Cache ─────────────────────────────────────────────────────────────

@dataclass
class MarketCache:
    markets: dict = field(default_factory=dict)   # series_ticker → [market_dict, ...]
    last_refresh: float = 0.0

    def get_best_market(self, series_ticker: str) -> Optional[dict]:
        """Return the highest-volume open market for a series."""
        markets = self.markets.get(series_ticker, [])
        if not markets:
            return None
        return max(markets, key=lambda m: float(m.get("volume", 0) or 0))


async def refresh_cache(http: httpx.AsyncClient, private_key, key_id: str, cache: MarketCache):
    all_series = list({s for _, s, _ in KEYWORD_MAP} | {"KXBTC", "KXETH", "KXFED", "KXCPI", "KXGDP"})
    for series in all_series:
        try:
            markets = await get_series_markets(http, private_key, key_id, series)
            cache.markets[series] = markets
        except Exception as e:
            log.warning(f"Cache refresh failed {series}: {e}")
    cache.last_refresh = asyncio.get_event_loop().time()
    log.info(f"Market cache refreshed: {sum(len(v) for v in cache.markets.values())} markets")


# ─── Claude Classification ────────────────────────────────────────────────────

async def classify_with_claude(claude_client: anthropic.Anthropic,
                                headline: str, description: str) -> Optional[dict]:
    """Returns {reprices, series_ticker, side, confidence, rationale} or None on error."""
    try:
        msg = claude_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            temperature=0,
            system=f"You are a prediction market analyst. Available Kalshi markets:\n{MARKET_CATALOG}\nRespond with JSON only. No explanation outside the JSON.",
            messages=[{"role": "user", "content":
                f"Does this news significantly reprice any Kalshi market?\n"
                f"HEADLINE: {headline}\nDESCRIPTION: {description[:300]}\n\n"
                f"If yes: {{\"reprices\":true,\"series_ticker\":\"KXBTC\",\"side\":\"yes\",\"confidence\":\"high\",\"rationale\":\"one sentence\"}}\n"
                f"If no: {{\"reprices\":false,\"series_ticker\":null,\"side\":null,\"confidence\":null,\"rationale\":\"not relevant\"}}\n"
                f"JSON:"}]
        )
        return json.loads(msg.content[0].text.strip())
    except Exception as e:
        log.warning(f"Claude classification failed: {e}")
        return None


# ─── RSS Polling ──────────────────────────────────────────────────────────────

async def poll_feed(feed: dict, seen_guids: set, news_queue: asyncio.Queue, http: httpx.AsyncClient):
    url = feed["url"]
    name = feed["name"]
    etag = None
    last_modified = None

    while True:
        try:
            headers = {}
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified

            r = await http.get(url, headers=headers, timeout=10.0)

            if r.status_code == 304:
                pass  # nothing new
            elif r.status_code == 200:
                etag = r.headers.get("etag")
                last_modified = r.headers.get("last-modified")

                root = ET.fromstring(r.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                items = root.findall(".//item") or root.findall(".//atom:entry", ns)

                new_count = 0
                for item in items:
                    guid = (item.findtext("guid") or item.findtext("atom:id", namespaces=ns) or
                            item.findtext("link") or "")
                    if guid and guid not in seen_guids:
                        seen_guids.add(guid)
                        title = item.findtext("title") or item.findtext("atom:title", namespaces=ns) or ""
                        desc = item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or ""
                        await news_queue.put({"headline": title.strip(), "description": desc.strip(),
                                             "source": name, "ts": datetime.now(timezone.utc).isoformat()})
                        new_count += 1

                if new_count:
                    log.info(f"[{name}] {new_count} new items")

        except Exception as e:
            log.warning(f"[{name}] Poll error: {e}")

        await asyncio.sleep(feed["poll_sec"])


# ─── Event Processor ─────────────────────────────────────────────────────────

async def process_events(news_queue: asyncio.Queue, http: httpx.AsyncClient,
                          private_key, key_id: str, claude_client: anthropic.Anthropic,
                          cache: MarketCache, paper_mode: bool, paper_balance: list):

    traded_headlines: set = set()  # avoid trading same headline twice

    while True:
        item = await news_queue.get()
        headline = item["headline"]
        description = item["description"]
        source = item["source"]

        if headline in traded_headlines:
            continue

        log.info(f"[{source}] {headline[:100]}")

        # ── Tier 1: keyword regex (fast path) ──────────────────────────────
        tier1_match = None
        for pattern, series, side in KEYWORD_MAP:
            if pattern.search(headline) or pattern.search(description[:200]):
                tier1_match = (series, side)
                break

        # ── Tier 2: Claude (parallel with Tier 1 execution if match) ───────
        claude_task = asyncio.create_task(classify_with_claude(claude_client, headline, description))

        if tier1_match:
            series, side = tier1_match
            log.info(f"Tier 1 match: {series} {side} from '{headline[:60]}'")
            market = cache.get_best_market(series)
            if market:
                await execute_trade(http, private_key, key_id, market, side,
                                    paper_mode, paper_balance, headline, "tier1")
                traded_headlines.add(headline)

        # Wait for Claude result
        claude_result = await claude_task
        if claude_result and claude_result.get("reprices") and claude_result.get("series_ticker"):
            series = claude_result["series_ticker"]
            side = claude_result.get("side", "yes")
            confidence = claude_result.get("confidence", "low")

            if confidence in ("high", "medium") and headline not in traded_headlines:
                log.info(f"Claude match: {series} {side} ({confidence}) — {claude_result.get('rationale','')}")
                market = cache.get_best_market(series)
                if market:
                    await execute_trade(http, private_key, key_id, market, side,
                                        paper_mode, paper_balance, headline, f"claude-{confidence}")
                    traded_headlines.add(headline)


async def execute_trade(http: httpx.AsyncClient, private_key, key_id: str,
                         market: dict, side: str, paper_mode: bool,
                         paper_balance: list, headline: str, tier: str):
    ticker = market["ticker"]

    # Get fresh orderbook
    ob = await get_orderbook(http, private_key, key_id, ticker)
    if not ob:
        return

    yes_bids = ob.get("orderbook_fp", {}).get("yes_dollars", [])
    no_bids = ob.get("orderbook_fp", {}).get("no_dollars", [])
    if not yes_bids or not no_bids:
        log.warning(f"Empty orderbook for {ticker}")
        return

    if side == "yes":
        # Take the best yes ask (cross the spread)
        best_no_bid = Decimal(no_bids[-1][0])
        price_cents = int((Decimal("1.0") - best_no_bid) * 100) + 1  # cross the ask
        price_cents = min(99, price_cents)
    else:
        best_yes_bid = Decimal(yes_bids[-1][0])
        price_cents = int((Decimal("1.0") - best_yes_bid) * 100) + 1
        price_cents = min(99, price_cents)

    # Fee-aware EV check: price must leave room for maker fee
    ev_after_fees = (100 - price_cents) / 100.0 - Config.MAKER_FEE
    if ev_after_fees <= 0:
        log.info(f"Skipping {ticker}: negative EV after {Config.MAKER_FEE*100}% fee (price={price_cents}¢)")
        shadow_log({"bot": "news", "ticker": ticker, "side": side, "price": price_cents, "tier": tier}, taken=False, reason="negative EV after fees")
        evaluate_virtual_portfolios({"bot": "news", "ticker": ticker, "side": side, "price": price_cents, "tier": tier})
        if _quant_modules_available:
            try:
                _features.extract({"price": locals().get("price", 0), "volume": locals().get("volume", 0), "bid": locals().get("bid", 0), "ask": locals().get("ask", 0)})
                _bayesian.update(locals().get("market_id", locals().get("ticker", "unknown")), locals().get("price", 0), time.time())
                _td_edge = calculate_time_weighted_edge(locals().get("edge", 0), locals().get("minutes_remaining", locals().get("time_remaining", 15)), 15)
                _vpin.update(locals().get("price", 0), locals().get("volume", 0))
                _mi = estimate_market_impact(locals().get("contracts", 1), locals().get("volume", 100))
            except:
                pass
        return

    # Kelly: 2% of balance per trade, capped at MAX_TRADE_USD
    price_dollars = price_cents / 100.0
    kelly_bet = min(paper_balance[0] * 0.02 * Config.KELLY_FRACTION, Config.MAX_TRADE_USD)
    count = max(1, int(kelly_bet / price_dollars))
    count = min(count, 200)

    # ── Risk Guard check ──
    if not paper_mode:
        allowed, reason, capped = risk_manager.pre_trade_check(ticker, price_cents, count, side, bot_name="news-bot")
        if not allowed:
            log.warning(f"Risk guard blocked: {reason}")
            return
        count = capped
    else:
        allowed, reason, capped = risk_manager.pre_trade_check(ticker, price_cents, count, side, bot_name="news-bot")
        if not allowed:
            log.info(f"[PAPER] Risk guard would block: {reason}")

    # ── Regime detection ──
    regime = check_regime(float(price_cents))
    if regime == "CRASH":
        log.warning("REGIME CRASH on kalshi_news_bot — skipping trade")
        shadow_log({"bot": "kalshi_news_bot", "regime": regime}, taken=False, reason="crash regime")
        evaluate_virtual_portfolios({"bot": "kalshi_news_bot", "regime": regime})
        return
    shadow_log({"bot": "news", "ticker": ticker, "side": side, "price": price_cents, "contracts": count, "tier": tier}, taken=True)
    evaluate_virtual_portfolios({"bot": "news", "ticker": ticker, "side": side, "price": price_cents, "contracts": count, "tier": tier})
    if paper_mode:
        cost = count * price_dollars
        paper_balance[0] -= cost
        log.info(f"[PAPER {tier.upper()}] {ticker} {side} @ {price_cents}¢ x{count} = ${cost:.2f} | balance: ${paper_balance[0]:.2f}")
        log.info(f"  Headline: {headline[:80]}")
    else:
        result = await place_order(http, private_key, key_id, ticker, side, price_cents, count, False)
        log.info(f"[LIVE {tier.upper()}] {ticker} {side} @ {price_cents}¢ x{count} | result: {result}")


# ─── Main ─────────────────────────────────────────────────────────────────────

# ── Stats HTTP server ─────────────────────────────────────────────────────────
_stats_app = Flask(__name__)
_bot_stats = {"trades": 0, "wins": 0, "pnl": 0.0, "balance": 0.0, "start": time.time()}

@_stats_app.route("/stats")
def _stats_endpoint():
    t = _bot_stats
    total = t["trades"]
    return jsonify({"bot": "kalshi-news-bot", "paper_mode": True,
        "balance": t["balance"], "trades": total, "wins": t["wins"],
        "losses": total - t["wins"], "win_rate": round(t["wins"]/max(total,1), 4),
        "pnl": t["pnl"], "uptime_hours": round((time.time()-t["start"])/3600, 2)})

@_stats_app.route("/health")
def _health_endpoint():
    return jsonify({"status": "ok"})

def _run_stats_server():
    _stats_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


async def main():
    private_key = load_private_key(Config.KALSHI_PRIVATE_KEY_PEM)
    claude_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    paper_balance = [Config.PAPER_BALANCE]
    _bot_stats["balance"] = paper_balance[0]
    threading.Thread(target=_run_stats_server, daemon=True).start()

    log.info(f"=== News Event Bot | {'PAPER' if Config.PAPER_MODE else 'LIVE'} MODE | balance=${paper_balance[0]:.2f} ===")

    news_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    seen_guids: set = set()
    cache = MarketCache()

    async with httpx.AsyncClient(http2=True, timeout=httpx.Timeout(10.0),
                                  limits=httpx.Limits(max_keepalive_connections=10)) as http:
        # Initial market cache
        await refresh_cache(http, private_key, Config.KALSHI_API_KEY_ID, cache)

        # Launch all tasks
        tasks = []

        # Feed pollers
        for feed in FEEDS:
            tasks.append(asyncio.create_task(
                poll_feed(feed, seen_guids, news_queue, http)))

        # Event processor
        tasks.append(asyncio.create_task(
            process_events(news_queue, http, private_key, Config.KALSHI_API_KEY_ID,
                           claude_client, cache, Config.PAPER_MODE, paper_balance)))

        # Cache refresh every 5 minutes
        async def cache_refresher():
            while True:
                await asyncio.sleep(300)
                await refresh_cache(http, private_key, Config.KALSHI_API_KEY_ID, cache)

        tasks.append(asyncio.create_task(cache_refresher()))

        # Status log every 5 minutes
        async def status_logger():
            while True:
                await asyncio.sleep(300)
                _bot_stats["balance"] = paper_balance[0]
                log.info(f"Status | queue={news_queue.qsize()} | seen={len(seen_guids)} | balance=${paper_balance[0]:.2f}")

        tasks.append(asyncio.create_task(status_logger()))

        log.info(f"Monitoring {len(FEEDS)} feeds | {len(KEYWORD_MAP)} keyword triggers")
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
