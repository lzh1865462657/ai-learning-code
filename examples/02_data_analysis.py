from pathlib import Path
from pprint import pprint

from ai_learning.data_analysis import load_sales, summarize_sales

project_root = Path(__file__).resolve().parents[1]
sales = load_sales(project_root / "data" / "sales.csv")
pprint(summarize_sales(sales), sort_dicts=False)
