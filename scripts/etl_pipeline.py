"""Amazon Product Sales — ETL Pipeline.

Mirrors notebooks/02_cleaning.ipynb so cleaning is reproducible from the CLI:

    python scripts/etl_pipeline.py

Reads:  data/raw/amazon_products_sales_data_raw.csv
Writes: data/processed/cleaned_data.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "amazon_products_sales_data_raw.csv"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "cleaned_data.csv"

NUMERIC_COLS = [
    "discounted_price",
    "original_price",
    "product_rating",
    "total_reviews",
    "purchased_last_month",
    "discount_percentage",
]
DATE_COLS = ["delivery_date", "data_collected_at"]
BOOL_LIKE_COLS = ["is_best_seller", "is_sponsored", "has_coupon"]
DROP_COLS = ["product_image_url", "product_page_url"]
REQUIRED_COLS = [
    "discounted_price",
    "original_price",
    "product_rating",
    "purchased_last_month",
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl")


def extract(path: Path) -> pd.DataFrame:
    log.info("Loading raw data from %s", path.relative_to(REPO_ROOT))
    df = pd.read_csv(path)
    log.info("Loaded shape: %s", df.shape)
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    log.info("Step 1 | Drop duplicates: %d -> %d rows", before, len(df))

    for col in NUMERIC_COLS:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    log.info("Step 2 | Coerced %d numeric columns", len(NUMERIC_COLS))

    for col in DATE_COLS:
        df.loc[:, col] = pd.to_datetime(df[col], errors="coerce")
    log.info("Step 3 | Parsed %d date columns", len(DATE_COLS))

    df.loc[:, "buy_box_availability"] = df["buy_box_availability"].fillna("No")
    log.info("Step 4 | Filled buy_box_availability nulls with 'No'")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    log.info("Step 5 | Dropped rows missing core fields: %d -> %d", before, len(df))

    before = len(df)
    df = df[df["discounted_price"] > 0]
    df = df[df["original_price"] > 0]
    df = df[(df["product_rating"] >= 0) & (df["product_rating"] <= 5)]
    df = df[df["discount_percentage"] <= 90]
    log.info(
        "Step 6 | Validated price/rating/discount ranges: %d -> %d", before, len(df)
    )

    df.loc[:, "product_category"] = (
        df["product_category"].astype(str).str.strip().str.lower()
    )
    log.info("Step 7 | Normalised product_category (strip + lowercase)")

    for col in BOOL_LIKE_COLS:
        df.loc[:, col] = df[col].astype(str).str.lower()
    log.info("Step 8 | Lowercased boolean-like columns: %s", BOOL_LIKE_COLS)

    df.loc[:, "estimated_revenue"] = (
        df["purchased_last_month"] * df["discounted_price"]
    )
    df.loc[:, "discount_value"] = df["original_price"] - df["discounted_price"]
    df.loc[:, "rating_category"] = np.where(df["product_rating"] >= 4, "High", "Low")
    log.info(
        "Step 9 | Engineered estimated_revenue, discount_value, rating_category"
    )

    df = df.drop(columns=DROP_COLS, errors="ignore")
    log.info("Step 10 | Dropped non-analytical columns: %s", DROP_COLS)

    before = len(df)
    df = df.dropna(subset=["purchased_last_month"])
    log.info(
        "Step 11 | Final dropna on purchased_last_month: %d -> %d", before, len(df)
    )

    return df


def load(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(
        "Wrote cleaned data to %s (shape=%s)",
        path.relative_to(REPO_ROOT),
        df.shape,
    )


def main() -> None:
    df = extract(RAW_PATH)
    df = transform(df)
    load(df, PROCESSED_PATH)
    log.info("ETL complete.")


if __name__ == "__main__":
    main()
