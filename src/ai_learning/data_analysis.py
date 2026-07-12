"""Sales data cleaning and aggregation examples with Pandas and NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "region", "product", "quantity", "unit_price"}


def load_sales(path: str | Path) -> pd.DataFrame:
    """Load sales records, validate their schema and calculate revenue."""

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"销售数据缺少字段: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="raise")
    frame["unit_price"] = pd.to_numeric(frame["unit_price"], errors="raise")
    if (frame[["quantity", "unit_price"]] < 0).any().any():
        raise ValueError("销量和单价不能为负数")

    frame["revenue"] = frame["quantity"] * frame["unit_price"]
    return frame


def summarize_sales(frame: pd.DataFrame) -> dict[str, object]:
    """Calculate headline metrics and grouped revenue rankings."""

    if frame.empty:
        raise ValueError("销售数据不能为空")
    if "revenue" not in frame:
        raise ValueError("请先使用 load_sales 计算 revenue 字段")

    region_revenue = (
        frame.groupby("region", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )
    product_revenue = (
        frame.groupby("product", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )
    revenue = frame["revenue"].to_numpy(dtype=float)

    return {
        "order_count": int(len(frame)),
        "total_revenue": round(float(np.sum(revenue)), 2),
        "average_order_value": round(float(np.mean(revenue)), 2),
        "top_region": region_revenue.iloc[0]["region"],
        "top_product": product_revenue.iloc[0]["product"],
        "region_revenue": region_revenue.to_dict(orient="records"),
    }
