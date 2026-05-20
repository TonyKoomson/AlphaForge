"""
Alpha Forge — Universe Fetcher
Fetches current stock universe lists from Wikipedia and saves them as CSV files
in data/universes/.

Usage:
    python tools/fetch_universe.py --name sp500
    python tools/fetch_universe.py --name nasdaq100
    python tools/fetch_universe.py --name all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd


UNIVERSES_DIR = ROOT / "data" / "universes"


def fetch_sp500() -> pd.DataFrame:
    """Fetch current S&P 500 components from Wikipedia."""
    import io, requests
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AlphaForge/1.0; research)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    # Wikipedia table columns vary slightly; normalise
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Common column name variants
    ticker_col = next(c for c in df.columns if "symbol" in c or "ticker" in c)
    name_col   = next((c for c in df.columns if "security" in c or "company" in c or "name" in c), None)
    sector_col = next((c for c in df.columns if "sector" in c or "gics_sector" in c), None)

    result = pd.DataFrame()
    result["ticker"]  = df[ticker_col].str.strip().str.replace(".", "-", regex=False)
    if name_col:
        result["name"] = df[name_col].str.strip()
    else:
        result["name"] = ""
    if sector_col:
        result["sector"] = df[sector_col].str.strip()
    else:
        result["sector"] = ""
    result["universe"] = "sp500"
    return result.dropna(subset=["ticker"]).reset_index(drop=True)


def fetch_nasdaq100() -> pd.DataFrame:
    """Fetch current NASDAQ 100 components from Wikipedia."""
    import io, requests
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AlphaForge/1.0; research)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    # Find the table that has a ticker/symbol column
    target = None
    for t in tables:
        cols = [c.lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in cols):
            target = t
            break
    if target is None:
        raise ValueError("Could not find NASDAQ 100 table on Wikipedia")

    target.columns = [c.strip().lower().replace(" ", "_") for c in target.columns]
    ticker_col = next(c for c in target.columns if "ticker" in c or "symbol" in c)
    name_col   = next((c for c in target.columns if "company" in c or "name" in c or "security" in c), None)

    result = pd.DataFrame()
    result["ticker"] = target[ticker_col].str.strip().str.replace(".", "-", regex=False)
    result["name"]   = target[name_col].str.strip() if name_col else ""
    result["sector"] = ""
    result["universe"] = "nasdaq100"
    return result.dropna(subset=["ticker"]).reset_index(drop=True)


def fetch_russell1000() -> pd.DataFrame:
    """Fetch Russell 1000 via iShares IWB holdings (most complete free source)."""
    # iShares IWB (Russell 1000 ETF) holdings CSV — publicly available
    url = "https://www.ishares.com/us/products/239707/ISHARES-RUSSELL-1000-ETF/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
    try:
        df = pd.read_csv(url, skiprows=9, thousands=",")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        ticker_col = next(c for c in df.columns if "ticker" in c or "symbol" in c)
        name_col   = next((c for c in df.columns if "name" in c), None)
        sector_col = next((c for c in df.columns if "sector" in c), None)

        result = pd.DataFrame()
        result["ticker"]  = df[ticker_col].str.strip()
        result["name"]    = df[name_col].str.strip() if name_col else ""
        result["sector"]  = df[sector_col].str.strip() if sector_col else ""
        result["universe"] = "russell1000"
        # Filter to actual equity tickers (skip cash, bonds etc.)
        result = result[result["ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
        return result.dropna(subset=["ticker"]).reset_index(drop=True)
    except Exception as e:
        print(f"  iShares fetch failed ({e}), falling back to Wikipedia S&P 500 + NASDAQ 100")
        sp500 = fetch_sp500()
        nq100 = fetch_nasdaq100()
        combined = pd.concat([sp500, nq100], ignore_index=True)
        combined["universe"] = "russell1000"
        return combined.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def fetch_russell2000() -> pd.DataFrame:
    """Fetch Russell 2000 via iShares IWM holdings (small-cap US equities)."""
    url = "https://www.ishares.com/us/products/239710/ISHARES-RUSSELL-2000-ETF/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    try:
        df = pd.read_csv(url, skiprows=9, thousands=",")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        ticker_col = next(c for c in df.columns if "ticker" in c or "symbol" in c)
        name_col   = next((c for c in df.columns if "name" in c), None)
        sector_col = next((c for c in df.columns if "sector" in c), None)

        result = pd.DataFrame()
        result["ticker"]  = df[ticker_col].str.strip()
        result["name"]    = df[name_col].str.strip() if name_col else ""
        result["sector"]  = df[sector_col].str.strip() if sector_col else ""
        result["universe"] = "russell2000"
        result = result[result["ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
        return result.dropna(subset=["ticker"]).reset_index(drop=True)
    except Exception as e:
        print(f"  iShares Russell 2000 fetch failed ({e}) — skipping")
        return pd.DataFrame(columns=["ticker", "name", "sector", "universe"])


def fetch_sector_etfs() -> pd.DataFrame:
    """Curated list of SPDR sector ETFs + major market ETFs."""
    rows = [
        # SPDR sector ETFs
        ("XLK",  "Technology Select Sector SPDR",        "Technology"),
        ("XLF",  "Financial Select Sector SPDR",         "Financials"),
        ("XLV",  "Health Care Select Sector SPDR",       "Health Care"),
        ("XLE",  "Energy Select Sector SPDR",            "Energy"),
        ("XLI",  "Industrial Select Sector SPDR",        "Industrials"),
        ("XLY",  "Consumer Discretionary Select SPDR",   "Cons. Discretionary"),
        ("XLP",  "Consumer Staples Select Sector SPDR",  "Cons. Staples"),
        ("XLU",  "Utilities Select Sector SPDR",         "Utilities"),
        ("XLB",  "Materials Select Sector SPDR",         "Materials"),
        ("XLRE", "Real Estate Select Sector SPDR",       "Real Estate"),
        ("XLC",  "Communication Services SPDR",          "Comm. Services"),
        # Major broad ETFs
        ("SPY",  "SPDR S&P 500",                         "Broad Market"),
        ("QQQ",  "Invesco QQQ (Nasdaq 100)",             "Broad Market"),
        ("IWM",  "iShares Russell 2000",                 "Small Cap"),
        ("IWB",  "iShares Russell 1000",                 "Large Cap"),
        ("VTI",  "Vanguard Total Stock Market",          "Broad Market"),
        ("DIA",  "SPDR Dow Jones Industrial",            "Broad Market"),
        # Fixed income
        ("TLT",  "iShares 20+ Year Treasury Bond",      "Bonds"),
        ("IEF",  "iShares 7-10 Year Treasury Bond",     "Bonds"),
        ("SHY",  "iShares 1-3 Year Treasury Bond",      "Short-Term Bonds"),
        ("LQD",  "iShares IG Corporate Bond",           "Corp Bonds"),
        ("HYG",  "iShares High Yield Corp Bond",        "High Yield"),
        # Commodities & alternatives
        ("GLD",  "SPDR Gold Shares",                    "Commodities"),
        ("SLV",  "iShares Silver Trust",                "Commodities"),
        ("USO",  "United States Oil Fund",              "Energy Commodities"),
        ("GDX",  "VanEck Gold Miners",                  "Gold Miners"),
        # International
        ("EFA",  "iShares MSCI EAFE",                   "International"),
        ("EEM",  "iShares MSCI Emerging Markets",       "Emerging Markets"),
        ("FXI",  "iShares China Large-Cap",             "China"),
        ("EWJ",  "iShares MSCI Japan",                  "Japan"),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "name", "sector"])
    df["universe"] = "etfs"
    return df


_FETCHERS = {
    "sp500":      fetch_sp500,
    "nasdaq100":  fetch_nasdaq100,
    "russell1000": fetch_russell1000,
    "russell2000": fetch_russell2000,
    "etfs":       fetch_sector_etfs,
}


def save_universe(name: str, verbose: bool = True) -> Path:
    """Fetch a universe and save it to data/universes/{name}.csv. Returns path."""
    UNIVERSES_DIR.mkdir(parents=True, exist_ok=True)
    if name not in _FETCHERS:
        raise ValueError(f"Unknown universe '{name}'. Available: {list(_FETCHERS)}")

    if verbose:
        print(f"Fetching {name} universe...", flush=True)

    df = _FETCHERS[name]()
    out_path = UNIVERSES_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)

    if verbose:
        print(f"  Saved {len(df)} tickers to {out_path}")
    return out_path


def load_universe(name_or_path: str) -> list[str]:
    """
    Load a list of tickers from a named universe or a CSV file path.

    Args:
        name_or_path: Universe name ('sp500', 'nasdaq100', 'etfs') OR a file path.

    Returns:
        List of ticker strings.
    """
    # Check if it's a file path
    p = Path(name_or_path)
    if not p.exists():
        p = UNIVERSES_DIR / f"{name_or_path}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Universe file not found: {p}\n"
            f"Run: python main.py fetch-universe --name {name_or_path}"
        )

    df = pd.read_csv(p)
    ticker_col = next(
        (c for c in df.columns if c.lower() in ("ticker", "symbol")),
        df.columns[0],
    )
    return [t.strip().upper() for t in df[ticker_col].dropna().tolist() if str(t).strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock universe lists")
    parser.add_argument("--name", default="sp500",
                        choices=list(_FETCHERS) + ["all"],
                        help="Universe to fetch")
    args = parser.parse_args()

    targets = list(_FETCHERS) if args.name == "all" else [args.name]
    for name in targets:
        try:
            save_universe(name)
        except Exception as e:
            print(f"  ERROR fetching {name}: {e}")
