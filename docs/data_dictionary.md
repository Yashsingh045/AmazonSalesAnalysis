# Data Dictionary — Amazon Product Sales Dataset

**Source file (raw):** `data/raw/amazon_products_sales_data_raw.csv`
**Source file (cleaned):** `data/processed/cleaned_data.csv`
**Cleaned shape:** 30,228 rows × 18 columns
**Collection window:** 2025-08-21 (single scrape snapshot, see `data_collected_at`)

The dataset captures product-level listings scraped from Amazon, covering pricing, ratings, sales velocity, badge status, and promotional flags across product categories. Each row is one product listing.

---

## Column reference

| # | Column | Type | Source | Description | Example |
|---|--------|------|--------|-------------|---------|
| 1 | `product_title` | string | raw | Full product name as displayed on Amazon. | `"Apple AirPods Pro 2 Wireless Earbuds, Active Noise Cancellation"` |
| 2 | `product_rating` | float | raw | Average customer star rating. Validation rule allows 0.0–5.0; observed range in cleaned data is 2.0–5.0. | `4.6` |
| 3 | `total_reviews` | float | raw | Total number of customer reviews on the listing. | `35882.0` |
| 4 | `purchased_last_month` | float | raw | Number of units sold in the past 30 days. Rows with null values are **dropped** during cleaning (step 11) — never imputed — to keep `estimated_revenue` fully populated. | `10000.0` |
| 5 | `discounted_price` | float | raw | Current selling price after any discount, in USD. | `162.24` |
| 6 | `original_price` | float | raw | Listed MSRP / pre-discount price, in USD. | `199.00` |
| 7 | `is_best_seller` | string | raw | Badge text shown on the listing. After cleaning, lowercased. Common values: `"no badge"`, `"best seller"`, `"amazon's"`, `"limited time deal"`, plus several `"save X%"` variants. **For analysis, treat `"best seller"` as the true badge.** | `"best seller"` |
| 8 | `is_sponsored` | string | raw | Whether the listing is paid placement. Cleaned to lowercase: `"sponsored"` or `"organic"`. | `"sponsored"` |
| 9 | `has_coupon` | string | raw | Coupon offer text, or `"no coupon"` if none. After cleaning, lowercased. **For analysis, treat anything other than `"no coupon"` as having a coupon.** | `"save 15% with coupon"` |
| 10 | `buy_box_availability` | string | raw | Whether the product is in the Amazon buy box. Missing values filled with `"No"` during cleaning. Common values: `"Add to cart"`, `"No"`. | `"Add to cart"` |
| 11 | `delivery_date` | string (ISO date) | raw | Estimated earliest delivery date shown to the customer. Stored as `YYYY-MM-DD` string in the CSV; consumers must call `pd.to_datetime(df['delivery_date'])` to use it as a datetime. 17.8% null (5,377 rows). | `"2025-09-01"` |
| 12 | `sustainability_tags` | string | raw | Sustainability / merchant badge text. Mostly null (92.1% missing). Most common values: `"Small Business"` (1,077), `"Carbon impact"` (661), `"Works with Alexa"` (372), `"Manufacturing practices"` (64). | `"Small Business"` |
| 13 | `data_collected_at` | string (ISO datetime) | raw | Timestamp when the row was scraped. Stored as `YYYY-MM-DD HH:MM:SS` string in the CSV; parse with `pd.to_datetime` on load. Effectively a single date for this snapshot. | `"2025-08-21 11:14:29"` |
| 14 | `product_category` | string | raw | Product category as classified by the scraper. Cleaned to lowercase + stripped. | `"phones"` |
| 15 | `discount_percentage` | float | raw | Pre-computed discount % off MSRP, scale 0–100. Validated to ≤ 90 during cleaning; observed range in cleaned data is 0.00–85.42. | `43.60` |
| 16 | `estimated_revenue` | float | **engineered** | `purchased_last_month × discounted_price`. Approximate monthly revenue per listing, in USD. **Primary KPI** for this project. | `1622400.00` |
| 17 | `discount_value` | float | **engineered** | `original_price - discounted_price`. Absolute dollars saved by the customer. | `36.76` |
| 18 | `rating_category` | string | **engineered** | `"High"` if `product_rating ≥ 4.0`, else `"Low"`. Used as a categorical predictor. | `"High"` |

---

## Cleaning rules applied (see `scripts/etl_pipeline.py` and `notebooks/02_cleaning.ipynb`)

1. **Drop duplicates** across all columns.
2. **Coerce numeric columns** (`discounted_price`, `original_price`, `product_rating`, `total_reviews`, `purchased_last_month`, `discount_percentage`) — invalid values become null.
3. **Parse dates** for `delivery_date` and `data_collected_at` in-memory during cleaning. (Note: pandas writes datetimes back to CSV as ISO-format strings, so the saved `cleaned_data.csv` stores them as strings — consumers must re-parse on load.)
4. **Fill `buy_box_availability`** nulls with `"No"`.
5. **Drop rows missing core fields:** `discounted_price`, `original_price`, `product_rating`, `purchased_last_month`.
6. **Validate ranges:** prices > 0, rating in [0, 5], discount_percentage ≤ 90.
7. **Normalise text:** lowercase + strip whitespace on `product_category`; lowercase boolean-like columns.
8. **Engineer KPIs:** `estimated_revenue`, `discount_value`, `rating_category`.
9. **Drop irrelevant columns:** `product_image_url`, `product_page_url` (URLs only, no analytical value).
10. **Final dropna** on `purchased_last_month` to ensure `estimated_revenue` is fully populated.

Net effect: **42,675 → 30,228 rows** (29% loss, predominantly due to missing sales counts).

---

## Limitations

- **Single snapshot:** all data collected on one date (2025-08-21). No trend / seasonality analysis possible.
- **Estimated revenue is an approximation.** Amazon publishes only `purchased_last_month` (rounded to 100s/1000s/10000s), so revenue estimates carry rounding error of up to ±5–10%.
- **Discount % is pre-computed** by the scraper and may diverge slightly from `(original - discounted) / original × 100` due to time-of-scrape promotions.
- **Boolean-like columns** (`is_best_seller`, `has_coupon`, `is_sponsored`) are stored as raw lowercased badge strings rather than booleans. Downstream analysis must derive 0/1 flags (e.g. in `notebooks/04_statistical_analysis.ipynb`, where `flag_best_seller`, `flag_sponsored`, and `flag_has_coupon` are derived in-notebook without modifying the CSV).
- **Date columns are written as strings**, not native datetime, after `df.to_csv(...)`. Always `pd.to_datetime()` `delivery_date` and `data_collected_at` after loading if you need date-based operations.
- **Sustainability tags** are too sparse (~8% present) for meaningful analysis as-is.
- **Categorical scope is narrow:** scraped categories are limited (e.g. Phones, Laptops, etc.) — the dataset is not representative of all Amazon products.
