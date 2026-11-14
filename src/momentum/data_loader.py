"""
Data foundation (roadmap Section 4), standalone-CSV variant.

The existing project loads prices from PostgreSQL (src/config/database.py,
src/data_collection/csv_importer.py). No PostgreSQL instance is available in
this environment, and the roadmap's raw per-symbol CSVs under
src/data_collection/raw/ already contain everything this first task needs
(date, OHLC, volume). This loader reads those CSVs directly, with the same
point-in-time discipline the roadmap requires: no forward-filled prices are
invented, no missing/no-trade session is converted into a genuine zero
return, and nothing from the future leaks into a given date's view.

If/when a PostgreSQL instance with the full `daily_prices` table is
available, this loader can be swapped for a query against that table without
changing anything downstream (everything past this module only consumes the
long-format panel returned by `load_price_panel`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "src" / "data_collection" / "raw"

# bt.csv / sfbt.csv use lowercase filenames but uppercase `symbole` columns;
# every other file is UPPERCASE.csv. Tunindex.csv is handled separately
# (different schema/format) by load_tunindex.
_STOCK_COLUMN_MAP = {
    "ouverture": "open",
    "haut": "high",
    "bas": "low",
    "cloture": "close",
    "volume": "volume",
    "date": "date",
    "symbole": "symbol",
}


def _load_one_stock_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = [c for c in _STOCK_COLUMN_MAP if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {missing}")
    df = raw.rename(columns=_STOCK_COLUMN_MAP)[
        ["symbol", "date", "open", "high", "low", "close", "volume"]
    ].copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    # Data-quality gate (roadmap 4.4): drop invalid/inconsistent OHLC and
    # non-positive prices instead of silently keeping them.
    valid = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
        & (df["volume"] >= 0)
    )
    dropped_invalid = (~valid).sum()
    df = df[valid]
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        df.attrs["dropped_rows"] = int(dropped)
        df.attrs["dropped_invalid_ohlc"] = int(dropped_invalid)
    return df


def load_price_panel(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load every per-symbol raw CSV (excluding Tunindex.csv) into one long panel.

    Returns columns: symbol, date, open, high, low, close, volume,
    sorted by (symbol, date). A session with no row for a symbol means that
    symbol did not trade that session -- it is never fabricated.
    """
    raw_dir = raw_dir or RAW_DIR
    frames = []
    quality_log = []
    for path in sorted(raw_dir.glob("*.csv")):
        if path.stem.lower() == "tunindex":
            continue
        try:
            df = _load_one_stock_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            quality_log.append({"file": path.name, "status": "FAILED", "error": str(exc)})
            continue
        if df.empty:
            quality_log.append({"file": path.name, "status": "EMPTY"})
            continue
        quality_log.append(
            {
                "file": path.name,
                "symbol": df["symbol"].iloc[0],
                "rows": len(df),
                "first_date": str(df["date"].min().date()),
                "last_date": str(df["date"].max().date()),
                "dropped_rows": int(df.attrs.get("dropped_rows", 0)),
            }
        )
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel.attrs["quality_log"] = quality_log
    return panel


def _parse_localized_number(value: str) -> float:
    text = str(value).strip().replace(" ", "").replace(" ", "")
    text = text.rstrip("%")
    if text.endswith("K"):
        text = text[:-1]
        mult = 1_000.0
    elif text.endswith("M"):
        text = text[:-1]
        mult = 1_000_000.0
    else:
        mult = 1.0
    if "," in text and "." in text and text.rfind(",") > text.rfind("."):
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text) * mult
    except ValueError:
        return float("nan")


def load_tunindex(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load TUNINDEX daily history (date, close), sorted ascending by date."""
    raw_dir = raw_dir or RAW_DIR
    path = raw_dir / "Tunindex.csv"
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    date_col = next(c for c in raw.columns if c.strip().lower() == "date")
    close_col = next(c for c in raw.columns if c.strip().lower() == "dernier")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce"),
            "close": raw[close_col].map(_parse_localized_number),
        }
    )
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    return out.sort_values("date").reset_index(drop=True)


def build_trading_calendar(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """The set of BVMT sessions used as the master calendar: the union of every
    session on which at least one symbol in the panel actually traded."""
    return pd.DatetimeIndex(sorted(panel["date"].unique()))


def build_close_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide symbol x date matrix of traded closes. NaN = no trade that session
    (never forward-filled here; forward-fill, where needed for valuation only,
    is done explicitly and documented at the point of use)."""
    return panel.pivot(index="date", columns="symbol", values="close")


def build_volume_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(index="date", columns="symbol", values="volume")


def build_open_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(index="date", columns="symbol", values="open")
