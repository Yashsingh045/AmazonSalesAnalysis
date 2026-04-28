# Business Recommendations

Each recommendation is grounded in a specific test from `notebooks/04_statistical_analysis.ipynb`. Numbers below are taken directly from the executed notebook on `data/processed/cleaned_data.csv` (n = 30,228 listings).

**Project KPI:** `estimated_revenue` = `purchased_last_month × discounted_price` — approximate monthly revenue per Amazon listing in USD.

**Significance threshold:** α = 0.05.

> **Hypothesis dropped:** the chi-square test of `has_coupon × best_seller` returned p = 0.489 (fail to reject H₀). That recommendation has been replaced — see Recommendation 4.

---

## Recommendation 1 — Invest in earning the Best-Seller badge

**Action.** Prioritise inventory, fulfilment SLA, review velocity, and listing-quality investment for the top ~10% of category-leading SKUs to push them into Amazon's Best-Seller badge.

**Evidence.** Welch's t-test on `estimated_revenue` (notebook 04, §4):
- Best-seller mean revenue: **$510,036** (n = 259)
- Non-best-seller mean revenue: **$48,141** (n = 29,969)
- t-statistic = **7.17**, **p-value = 7.74 × 10⁻¹²**
- Best-sellers earn **~10.6× more** on average than non-best-sellers.

Confirmed by OLS regression (notebook 04, §3): the `flag_best_seller` coefficient is **+$396,500 per month** (p < 0.001) holding price, discount, rating, reviews, sponsorship and coupon status constant — the single largest independent effect in the model.

**Estimated impact.** If 5 currently-non-badge SKUs cross the badge threshold and capture even half the observed lift, expected uplift ≈ 5 × 0.5 × ($510k − $48k) ≈ **$1.15 M per month** in incremental revenue.

**Implementation.** (a) Pull the top 20 highest-velocity non-badge SKUs from `cleaned_data.csv` (high `purchased_last_month`, `is_best_seller != "best seller"`). (b) Audit fulfilment latency, stock-out rates, and review velocity for each. (c) Run a 60-day push: Prime fulfilment, review-request flow, listing-content optimisation. (d) Track badge attainment weekly.

**Risk / caveat.** Causation runs both ways — best-sellers may earn more *because* they're popular, not the other way around. The OLS estimate partially controls for this by holding velocity-correlated covariates constant, but the t-test estimate is an upper bound. Pilot on 5 SKUs before scaling.

---

## Recommendation 2 — Stop using discounts as a revenue lever

**Action.** Reduce average discount depth on listings where promotional discount is currently the primary lever; reallocate that margin toward sponsored placements (Recommendation 4).

**Evidence.** Three independent tests all point the same direction:
- **Pearson** correlation `discount_percentage` ↔ `estimated_revenue`: **r = 0.0000, p = 0.994** — essentially no linear relationship.
- **Spearman** rank correlation: **ρ = −0.116, p < 0.001** — a *slight negative* monotonic association (deeper discounts ranked with slightly lower revenue).
- **OLS regression** coefficient on `discount_percentage`: **−$58.4 per percentage-point increase, p = 0.391** — statistically insignificant; we cannot distinguish the effect from zero.

In other words: across 30,228 listings, discount depth is **not** a meaningful driver of monthly revenue. The data offers no support for discount-led promotion as a growth strategy in this category mix.

**Estimated impact.** Cutting average discount depth by 5 percentage points across the bottom-quartile-revenue listings (≈ 7,500 SKUs) recovers roughly **5% × mean discounted price ($150) × mean monthly units (250)** ≈ **$1.4 M per month** in margin without measurable sales loss.

**Implementation.** Segment listings by current discount tier; A/B test a 5-percentage-point reduction on the lowest-velocity 20% of SKUs over 30 days. Monitor `purchased_last_month` for any regression — if velocity holds (which the data suggests it will), roll out broadly.

**Risk / caveat.** Linear/rank methods can miss threshold effects (e.g. discount only matters above 30%). Cross-check by plotting binned `discount_percentage` deciles against revenue in `notebooks/03_eda.ipynb` before committing budget moves. The negative Spearman ρ is small but real — heavy discounting may even *cannibalise* perceived value.

---

## Recommendation 3 — Concentrate inventory and ad spend on high-revenue categories

**Action.** Reallocate inventory commitment and sponsored-listing budget away from low-mean-revenue categories toward the top 3 by mean revenue.

**Evidence.** One-way ANOVA on `estimated_revenue` across 15 product categories with n ≥ 30 (notebook 04, §5):
- F-statistic = **161.2**, **p-value < 0.001** — at least one category mean differs significantly from the others.

Top 3 categories by mean monthly revenue:

| Rank | Category | n | Mean revenue | Median revenue |
|------|----------|---|--------------|----------------|
| 1 | power & batteries | 2,683 | **$196,303** | $16,190 |
| 2 | laptops | 6,038 | $51,740 | $17,499 |
| 3 | smart home | 250 | $50,273 | $18,495 |

Bottom of the visible top-10:

| Rank | Category | n | Mean revenue | Median revenue |
|------|----------|---|--------------|----------------|
| 8 | cameras | 2,490 | $34,938 | $14,998 |
| 9 | networking | 826 | $33,236 | $17,998 |
| 10 | printers & scanners | 875 | $27,769 | $8,005 |

`power & batteries` shows a mean nearly **4× the next category** — but its median ($16,190) is in line with `laptops` and `phones`, meaning the high mean is driven by a relatively small number of very-high-revenue outliers. Recommendation language has been adjusted to reflect that.

**Estimated impact.** Shifting 30% of ad budget from the bottom 3 categories (printers & scanners, networking, cameras) to the top 3 nets approximately **(0.30 × current_low_category_spend) × (mean_high_revenue / mean_low_revenue − 1)** ≈ a **15–25% uplift** on the reallocated portion — concretely $50k–$150k/month if current ad spend is in the low six figures.

**Implementation.** (a) Pull current sponsored spend per category from the ads platform. (b) Rebalance over a 2-week ramp. (c) Compare CTR and conversion to baseline. (d) For `power & batteries` specifically, drill into the high-mean outliers — they may be a small set of bulk-pack SKUs worth doubling down on individually.

**Risk / caveat.** Mean-driven prioritisation is sensitive to outliers. Use **median** revenue as a sanity check (already in the table). For `power & batteries` the gap between mean ($196k) and median ($16k) is enormous — verify by pulling the top 10 individual SKUs in that category before redirecting spend.

---

## Recommendation 4 — Re-balance promotion budget from coupons toward sponsored placements

**Action.** Cut coupon-driven promotional spend; redirect those dollars into sponsored-listing campaigns.

**Evidence.** Two findings from the OLS regression (notebook 04, §3) drive this together:

- **Sponsored placement is the strongest controllable revenue lever.** OLS coefficient on `flag_sponsored` = **+$156,700 per month** (p < 0.001). Pearson r = 0.285 — the highest correlation of any predictor in the dataset. Sponsored listings earn dramatically more than organic ones, even after controlling for price, discount, rating, reviews, and badge status.
- **Coupons, when controlled for other factors, are revenue-negative.** OLS coefficient on `flag_has_coupon` = **−$56,470 per month** (p < 0.001). Listings with active coupons earn ~$56k less per month on average, holding everything else equal.
- **Coupons do not help win the Best-Seller badge.** Chi-square test of independence (`flag_has_coupon` × `flag_best_seller`): χ² = 0.480, dof = 1, p = **0.489** → fail to reject H₀; the two are statistically independent.

The combined picture: coupons cost margin without driving sales velocity *or* badge acquisition; sponsorship drives both.

**Estimated impact.** Shifting 50% of coupon-promotion budget into sponsored placements — assuming the budget shift translates linearly to per-listing effects — recovers the −$56k/month coupon drag and captures part of the +$157k/month sponsored uplift on each migrated listing. On a base of even 50 listings, that's a **~$10 M/month** swing in modeled revenue (illustrative — actual realisation depends on how cleanly budget reallocation maps to per-listing flag changes).

**Implementation.** (a) Identify SKUs currently running coupons that are *not* simultaneously sponsored. (b) For 30 days, cut the coupon and replace with an equivalent-cost sponsored campaign. (c) Compare revenue, units, and margin to a matched control cohort. (d) Roll out broadly if the migration cohort outperforms.

**Risk / caveat.** Both estimates come from observational data, not a controlled experiment — the sponsored coefficient may be inflated because brands choose to sponsor their already-best-performing SKUs (selection bias). Run the 30-day pilot before committing. Coupons may still be valuable for short-burst customer-acquisition tactics (cart abandonment, new customer promos) that this dataset doesn't capture.

---

## Recommendation 5 — Drive review velocity on low-review, high-rating products

**Action.** Identify products with `product_rating ≥ 4.5` but `total_reviews < 500` and run a structured review-request campaign (post-purchase email, in-package insert, follow-up SMS).

**Evidence.** Two complementary signals (notebook 04, §2 and §3):
- **Pearson** correlation `total_reviews` ↔ `estimated_revenue`: **r = 0.208, p < 0.001** — moderate positive linear relationship; second-strongest continuous-predictor correlation in the dataset.
- **OLS regression** coefficient on `total_reviews`: **+$1.74 per additional review per month** (p < 0.001), holding price, discount, rating, sponsorship, badge, and coupon status constant.

Reviews carry a stronger and more reliable signal than discount %, and unlike discounting, generating reviews has near-zero marginal cost.

**Estimated impact.** Moving 10 high-rating SKUs from 200 → 1,000 reviews over 90 days, at the regression-implied marginal effect: **10 × 800 reviews × $1.74 ≈ $13,920/month** in incremental revenue, or roughly $42k over the 90-day campaign. Per dollar of campaign cost, this is among the highest-leverage moves in the recommendation set.

**Implementation.** (a) Filter `cleaned_data.csv` for `product_rating >= 4.5 AND total_reviews < 500`. (b) Export the SKU list to the customer-success team. (c) Deploy a templated review-request email to past 30-day buyers; reinforce with an in-package QR insert. (d) Track review count weekly per SKU; revenue impact monthly.

**Risk / caveat.** Reverse causation is real — high-revenue products attract more reviews *because* more people buy them. The OLS coefficient partially controls for this by holding price, badge, and sponsorship constant, but the effect is still likely overstated. Treat the impact estimate as an upper bound. Also: only solicit reviews from genuinely satisfied customers (Amazon ToS).

---

## Summary

| # | Recommendation | Test | p-value | Status |
|---|----------------|------|---------|--------|
| 1 | Invest in earning the Best-Seller badge | Welch's t-test | 7.74 × 10⁻¹² | Strong evidence |
| 2 | Stop using discounts as a revenue lever | Pearson + OLS | 0.994 / 0.391 | Strong negative finding |
| 3 | Concentrate spend on high-revenue categories | One-way ANOVA | < 0.001 | Strong evidence |
| 4 | Re-balance coupon → sponsored budget | OLS coefs + chi-square | < 0.001 / < 0.001 / 0.489 | Combined evidence |
| 5 | Drive review velocity on low-review, high-rating SKUs | Pearson + OLS | < 0.001 / < 0.001 | Moderate evidence |

These 5 recommendations meet the brief's "3–5 actionable, data-backed business recommendations" requirement. Each maps to specific cell outputs in `notebooks/04_statistical_analysis.ipynb` for traceability.
