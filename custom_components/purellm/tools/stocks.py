"""Stock price tool handler."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from ..utils.http_client import fetch_json, log_and_error

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Common company name to symbol mappings
COMPANY_TO_SYMBOL = {
    "apple": "AAPL", "tesla": "TSLA", "google": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT", "amazon": "AMZN", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "disney": "DIS", "nike": "NKE",
    "coca-cola": "KO", "coke": "KO", "pepsi": "PEP", "walmart": "WMT",
    "costco": "COST", "starbucks": "SBUX", "mcdonalds": "MCD", "boeing": "BA",
    "intel": "INTC", "amd": "AMD", "paypal": "PYPL", "visa": "V",
    "mastercard": "MA", "jpmorgan": "JPM", "goldman": "GS", "berkshire": "BRK-B",
    "johnson": "JNJ", "pfizer": "PFE", "moderna": "MRNA", "uber": "UBER",
    "lyft": "LYFT", "airbnb": "ABNB", "spotify": "SPOT", "snap": "SNAP",
    "twitter": "X", "x": "X", "salesforce": "CRM", "oracle": "ORCL",
    "ibm": "IBM", "cisco": "CSCO", "adobe": "ADBE", "zoom": "ZM",
}


async def get_stock_price(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    track_api_call: callable,
) -> dict[str, Any]:
    """Get stock price from Yahoo Finance API (free, no key required).

    Args:
        arguments: Tool arguments (symbol)
        session: aiohttp session
        track_api_call: Callback to track API usage

    Returns:
        Stock price data dict
    """
    symbol = arguments.get("symbol", "").upper().strip()

    if not symbol:
        return {"error": "No stock symbol provided"}

    # Convert company name to symbol if needed
    symbol_lookup = symbol.lower()
    if symbol_lookup in COMPANY_TO_SYMBOL:
        symbol = COMPANY_TO_SYMBOL[symbol_lookup]

    try:
        track_api_call("stocks")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"

        data, status = await fetch_json(session, url, headers=headers)
        if data is None:
            return {"error": f"Could not find stock symbol '{symbol}'"}

        result_data = data.get("chart", {}).get("result", [{}])[0]
        meta = result_data.get("meta", {})

        if not meta or "regularMarketPrice" not in meta:
            return {"error": f"Could not find stock data for '{symbol}'"}

        price = meta.get("regularMarketPrice", 0)
        prev_close = meta.get("previousClose", meta.get("chartPreviousClose", price))
        change = price - prev_close if prev_close else 0
        pct_change = (change / prev_close * 100) if prev_close else 0
        company_name = meta.get("shortName", meta.get("longName", symbol))

        direction = "up" if change >= 0 else "down"
        result = {
            "symbol": symbol,
            "company": company_name,
            "price": round(price, 2),
            "change": round(change, 2),
            "percent_change": round(pct_change, 2),
            "previous_close": round(prev_close, 2) if prev_close else None,
            "response_text": f"{company_name} ({symbol}) is at ${price:.2f}, {direction} ${abs(change):.2f} ({pct_change:+.2f}%) today."
        }

        _LOGGER.info("Stock price for %s: %s", symbol, result.get("response_text", ""))
        return result

    except Exception as err:
        return log_and_error("Failed to get stock price", err)
