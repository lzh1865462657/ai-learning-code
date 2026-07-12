from pathlib import Path

import pytest

from ai_learning.data_analysis import load_sales, summarize_sales

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sales_summary() -> None:
    sales = load_sales(PROJECT_ROOT / "data" / "sales.csv")
    summary = summarize_sales(sales)

    assert summary["order_count"] == 8
    assert summary["total_revenue"] == 10174
    assert summary["top_region"] == "华南"
    assert summary["top_product"] == "Python 训练营"


def test_load_sales_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("date,region\n2026-01-01,华东\n", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少字段"):
        load_sales(path)
