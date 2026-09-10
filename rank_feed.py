"""
Ranks marketplace listings for the default homepage feed.

Reads the snapshot and returns the top N listings (default 20) to show a user
who has not searched for anything yet.

Each listing is scored in expected dollars per impression:

    score = price * shrunk_conversion_rate * quality_multiplier * freshness_multiplier

The first two factors give the expected revenue from showing the listing once.
The quality multiplier discounts revenue from listings people rate badly, and
the freshness multiplier is a small boost for new listings. Slots are then
filled with a reserved exploration budget and a cap on how much of the feed any
one price quartile can take. README.md sets out the reasoning for each piece.

Usage:

    python rank_feed.py                    # print and write the top 20
    python rank_feed.py --top 50 -o feed.csv

Requires: pandas, numpy.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Business parameters.
#
# These are the settings worth arguing about in a product review, so they live
# in one place rather than being scattered through the code. Values are either
# fitted from the data or chosen for reasons set out in the README.
# --------------------------------------------------------------------------


@dataclass
class Config:
    # --- Conversion model -------------------------------------------------
    # Price buckets used to estimate the shrinkage strength. Enough of them to
    # see how spread varies across the price ladder, each still large enough
    # for its variance estimate to be stable.
    n_price_buckets: int = 20

    # --- Quality (rating) model -------------------------------------------
    # Pseudo-count for the Bayesian average rating, i.e. how many reviews a
    # listing needs before its own rating outweighs the platform mean. The
    # median listing has 6 reviews and the 75th percentile has 18, so at 10 a
    # typical listing sits about halfway between the two.
    rating_prior_weight: float = 10.0

    # Score scales with (rating / prior_rating) ** quality_gamma. At gamma = 0
    # ratings are ignored; at gamma = 2 a 2.1-star listing is worth about a
    # quarter of an average one per impression. This is a business judgement
    # rather than a fitted parameter. Rating does not predict conversion in
    # this data, so the case for it rests on costs the file does not contain:
    # refunds, support load and churn.
    quality_gamma: float = 2.0

    # A listing whose shrunk rating falls below this is kept off the feed
    # whatever it earns. Three listings hit this in the current snapshot.
    hard_quality_floor: float = 3.0

    # --- Freshness / exploration -----------------------------------------
    # Bounded boost for brand-new listings: 1 + boost * exp(-days / tau).
    freshness_boost: float = 0.15
    freshness_tau_days: float = 30.0

    # Share of the feed reserved for under-measured listings. A slot budget
    # caps the cost up front, which a score bonus would not.
    exploration_slot_share: float = 0.15

    # --- Slate composition ------------------------------------------------
    # No catalogue price quartile may take more than this share of the feed.
    # Revenue per impression is roughly flat across the price ladder, so this
    # costs about 1.3% and gives a first-time visitor a range to shop from.
    max_share_per_price_quartile: float = 0.40

    # --- Data hygiene -----------------------------------------------------
    # average_rating is documented as 1.0-5.0, but 293 listings carry 0.0 and
    # all of them have zero reviews. 0.0 means "not rated yet" rather than a
    # zero-star score, so it is treated as missing.
    rating_missing_sentinel: float = 0.0

    required_columns: Tuple[str, ...] = field(
        default=(
            "item_id",
            "price",
            "days_on_platform",
            "historical_views",
            "historical_purchases",
            "average_rating",
            "review_count",
        )
    )


# --------------------------------------------------------------------------
# Load and validate
# --------------------------------------------------------------------------


def load_listings(path: str, cfg: Config) -> pd.DataFrame:
    """Read the snapshot and repair or flag anything that would break the maths.

    Changes are recorded on the frame and reported with the slate rather than
    being applied silently.
    """
    df = pd.read_csv(path)

    missing = [c for c in cfg.required_columns if c not in df.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (path, missing))

    notes = []

    dupes = int(df["item_id"].duplicated().sum())
    if dupes:
        notes.append("dropped %d duplicate item_id rows" % dupes)
        df = df.drop_duplicates("item_id", keep="first")

    numeric = [c for c in cfg.required_columns if c != "item_id"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    bad = df[numeric].isna().any(axis=1)
    if bad.any():
        notes.append("dropped %d rows with non-numeric or missing fields" % int(bad.sum()))
        df = df.loc[~bad]

    # Purchases cannot exceed impressions. If the log says otherwise, clip it
    # rather than let a broken ratio take a slot.
    over = df["historical_purchases"] > df["historical_views"]
    if over.any():
        notes.append("clipped %d rows with purchases > views" % int(over.sum()))
        df.loc[over, "historical_purchases"] = df.loc[over, "historical_views"]

    nonneg = [
        "price",
        "days_on_platform",
        "historical_views",
        "historical_purchases",
        "review_count",
    ]
    df[nonneg] = df[nonneg].clip(lower=0)
    df["days_on_platform"] = df["days_on_platform"].clip(lower=1)
    # A zero price would break the revenue model, so treat it as a data error.
    df["price"] = df["price"].clip(lower=0.01)

    # 0.0 with no reviews means unrated, not badly rated.
    df["is_rated"] = (df["review_count"] > 0) & (
        df["average_rating"] > cfg.rating_missing_sentinel
    )
    n_unrated = int((~df["is_rated"]).sum())
    if n_unrated:
        notes.append(
            "treated %d listings with no reviews as unrated rather than 0-star" % n_unrated
        )

    df = df.reset_index(drop=True)
    df.attrs["load_notes"] = notes
    return df


# --------------------------------------------------------------------------
# 1. Conversion: empirical-Bayes shrinkage toward a price-conditional prior
# --------------------------------------------------------------------------


def _n_buckets(n_rows: int, cfg: Config) -> int:
    """Cap the bucket count so small inputs do not produce empty buckets."""
    return int(min(cfg.n_price_buckets, max(2, n_rows // 10)))


def fit_revenue_per_impression(df: pd.DataFrame) -> float:
    """Pooled revenue per impression, used as the conversion prior.

    Conversion falls roughly as fast as price rises, so price * conversion is
    close to constant across the catalogue. Fitting log(cvr) = a + b*log(price)
    gives b = -0.895 with a weighted R^2 of 0.963. That is near enough to -1
    that I assume the two cancel exactly, which reduces the prior to a single
    number. On held-out traffic the fitted curve scored $2.810 per impression
    against $2.818 for this version, so simplifying costs nothing.

    A $220 listing is therefore expected to convert at R / 220 before any of
    its own traffic is taken into account.
    """
    views = df["historical_views"].sum()
    if views <= 0:
        return 1.0
    return float((df["price"] * df["historical_purchases"]).sum() / views)


def fit_shrinkage_strength(df: pd.DataFrame, prior: pd.Series, cfg: Config) -> float:
    """Estimate the prior's weight in pseudo-purchases by method of moments.

    Within each price bucket the observed spread splits into two parts:

        Var(observed cvr around the prior) = Var(true cvr) + E[ p(1-p) / n ]

    Subtracting the binomial sampling noise leaves the genuine spread between
    listings, and the smaller that is, the harder we should shrink. I take the
    median across buckets rather than the mean because one bucket here has its
    variance inflated by a single outlier.
    """
    measured = df[df["historical_views"] > 0].copy()
    measured["_prior"] = prior.loc[measured.index]
    measured["_bucket"] = pd.qcut(
        measured["price"].rank(method="first"), _n_buckets(len(measured), cfg), labels=False
    )

    alphas = []
    for _, x in measured.groupby("_bucket"):
        cvr = x["historical_purchases"] / x["historical_views"]
        p = float(np.average(x["_prior"], weights=x["historical_views"]))
        resid_var = float(((cvr - x["_prior"]) ** 2).mean())
        noise = float((cvr * (1 - cvr) / x["historical_views"]).mean())
        true_var = resid_var - noise
        if true_var > 1e-9:
            strength = p * (1 - p) / true_var - 1.0
            if strength > 0:
                alphas.append(p * strength)

    alpha0 = float(np.median(alphas)) if alphas else 4.0
    return float(np.clip(alpha0, 1.0, 20.0))


def add_conversion_estimate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    revenue_per_impression = fit_revenue_per_impression(df)

    df = df.copy()
    # Prior conversion rate for a listing at this price: R / price.
    df["prior_cvr"] = np.clip(revenue_per_impression / df["price"], 1e-5, 0.99)

    alpha0 = fit_shrinkage_strength(df, df["prior_cvr"], cfg)

    # Beta(alpha, beta) prior with mean = prior_cvr and alpha = alpha0
    # pseudo-purchases, which makes its weight in impressions alpha0/prior_cvr.
    #
    # Fixing a pseudo-purchase count rather than a pseudo-view count means the
    # prior counts for more impressions where conversion is rarer. Measuring a
    # 1.5% rate as precisely as an 18% rate needs about twelve times the
    # traffic, and alpha0/p scales that way on its own.
    df["prior_views"] = alpha0 / df["prior_cvr"]

    a = df["historical_purchases"] + alpha0
    b = df["prior_views"] - alpha0 + (df["historical_views"] - df["historical_purchases"])

    df["raw_cvr"] = np.where(
        df["historical_views"] > 0,
        df["historical_purchases"] / df["historical_views"].replace(0, np.nan),
        np.nan,
    )
    # Posterior mean of the Beta-Binomial, i.e. the shrunk conversion rate.
    # This is the right point estimate when the objective is expected revenue.
    df["cvr"] = a / (a + b)

    # A listing counts as under-measured when it has had fewer impressions than
    # its own prior is worth, meaning its own data has not yet outweighed the
    # assumption we started from.
    df["is_under_measured"] = df["historical_views"] < df["prior_views"]

    df.attrs["alpha0"] = alpha0
    df.attrs["revenue_per_impression"] = revenue_per_impression
    return df


# --------------------------------------------------------------------------
# 2. Quality: Bayesian average rating, applied as a multiplier
# --------------------------------------------------------------------------


def add_quality_multiplier(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    rated = df[df["is_rated"]]
    prior_rating = float(rated["average_rating"].mean()) if len(rated) else 4.0

    # Bayesian average, so a listing with two reviews stays close to the
    # platform mean. Unrated listings land on the mean exactly.
    obs = df["average_rating"].where(df["is_rated"], prior_rating)
    n = df["review_count"].where(df["is_rated"], 0)
    df["rating"] = (obs * n + prior_rating * cfg.rating_prior_weight) / (
        n + cfg.rating_prior_weight
    )

    # Multiplicative rather than additive, so the discount stays proportional
    # at every revenue level. An additive penalty would barely dent a listing
    # earning $5 an impression while wiping out one earning $1.
    df["quality_multiplier"] = (df["rating"] / prior_rating) ** cfg.quality_gamma

    # Applied to the shrunk rating, so one bad review cannot delist a listing
    # that is otherwise fine.
    df["below_quality_floor"] = df["rating"] < cfg.hard_quality_floor

    df.attrs["prior_rating"] = prior_rating
    return df


# --------------------------------------------------------------------------
# 3. Freshness
# --------------------------------------------------------------------------


def add_freshness_multiplier(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    # Conversion is flat across age deciles in this snapshot, so age tracks how
    # much a listing has been shown rather than how good it is. This term is
    # not a relevance correction, it is a subsidy for new supply, which is why
    # it is capped at 15% and decays within a month.
    df["freshness_multiplier"] = 1.0 + cfg.freshness_boost * np.exp(
        -df["days_on_platform"] / cfg.freshness_tau_days
    )
    return df


# --------------------------------------------------------------------------
# 4. Score and slate construction
# --------------------------------------------------------------------------


def score_listings(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_conversion_estimate(df, cfg)
    df = add_quality_multiplier(df, cfg)
    df = add_freshness_multiplier(df, cfg)

    # Expected revenue from showing this listing once, which is what a homepage
    # slot actually spends.
    df["exp_revenue_per_impression"] = df["price"] * df["cvr"]
    df["score"] = (
        df["exp_revenue_per_impression"]
        * df["quality_multiplier"]
        * df["freshness_multiplier"]
    )
    df["eligible"] = ~df["below_quality_floor"]

    # Catalogue price quartile, used by the slate diversity cap.
    df["price_quartile"] = pd.qcut(
        df["price"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)

    # Tie-break on item_id so equal scores always resolve the same way and the
    # feed does not reshuffle between page loads.
    return df.sort_values(["score", "item_id"], ascending=[False, True]).reset_index(drop=True)


def _greedy_fill(pool: pd.DataFrame, n: int, counts: dict, cap: int, taken: set) -> list:
    """Take the n highest-scoring rows that keep each price quartile under cap."""
    picked = []
    for _, row in pool.iterrows():
        if len(picked) >= n:
            break
        if row["item_id"] in taken:
            continue
        q = row["price_quartile"]
        if counts.get(q, 0) >= cap:
            continue
        picked.append(row)
        counts[q] = counts.get(q, 0) + 1
        taken.add(row["item_id"])
    return picked


def build_slate(df: pd.DataFrame, cfg: Config, top_n: int = 20) -> pd.DataFrame:
    """Fill the feed with proven listings first, then the reserved exploration
    slots, keeping each price quartile under its cap."""
    pool = df[df["eligible"]]
    n_explore = int(round(top_n * cfg.exploration_slot_share))
    cap = max(1, int(np.ceil(top_n * cfg.max_share_per_price_quartile)))

    counts: dict = {}
    taken: set = set()

    exploit = _greedy_fill(
        pool[~pool["is_under_measured"]], top_n - n_explore, counts, cap, taken
    )
    explore = _greedy_fill(pool[pool["is_under_measured"]], n_explore, counts, cap, taken)

    rows = exploit + explore
    # Backfill from the whole eligible pool, respecting the cap first and then
    # ignoring it, so the slate always comes back full.
    if len(rows) < top_n:
        rows += _greedy_fill(pool, top_n - len(rows), counts, cap, taken)
    if len(rows) < top_n:
        rows += _greedy_fill(pool, top_n - len(rows), counts, top_n, taken)

    if not rows:
        raise ValueError("no listings passed the eligibility filters")

    slate = pd.DataFrame(rows)
    slate["slot_type"] = np.where(slate["is_under_measured"], "explore", "exploit")
    slate = slate.sort_values(["score", "item_id"], ascending=[False, True])
    return slate.head(top_n).reset_index(drop=True).assign(rank=lambda d: d.index + 1)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

DISPLAY = [
    "rank",
    "item_id",
    "slot_type",
    "score",
    "exp_revenue_per_impression",
    "price",
    "price_quartile",
    "cvr",
    "raw_cvr",
    "rating",
    "average_rating",
    "review_count",
    "historical_views",
    "historical_purchases",
    "days_on_platform",
]


def print_slate(slate: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> None:
    out = slate[DISPLAY].copy()
    out["score"] = out["score"].round(3)
    out["exp_revenue_per_impression"] = out["exp_revenue_per_impression"].round(3)
    out["cvr"] = (out["cvr"] * 100).round(2)
    out["raw_cvr"] = (out["raw_cvr"] * 100).round(2)
    out["rating"] = out["rating"].round(2)
    out = out.rename(
        columns={
            "score": "score($/imp)",
            "exp_revenue_per_impression": "erpi($)",
            "price_quartile": "pq",
            "cvr": "cvr%",
            "raw_cvr": "raw_cvr%",
            "rating": "rating_adj",
            "average_rating": "rating_raw",
            "review_count": "reviews",
            "historical_views": "views",
            "historical_purchases": "purch",
            "days_on_platform": "age_d",
        }
    )
    print("\nTOP %d - DEFAULT HOMEPAGE FEED" % len(out))
    print("=" * 142)
    print(out.to_string(index=False))
    print("=" * 142)

    n_ex = int((slate["slot_type"] == "explore").sum())
    mix = slate["price_quartile"].value_counts().reindex(["Q1", "Q2", "Q3", "Q4"]).fillna(0)
    print(
        "\nSlate: %d proven + %d exploration | price mix Q1/Q2/Q3/Q4 = %d/%d/%d/%d "
        "($%.2f-$%.2f) | mean expected revenue/impression $%.2f vs platform median $%.2f "
        "| mean adjusted rating %.2f vs platform %.2f | median age %.0fd"
        % (
            len(slate) - n_ex,
            n_ex,
            mix["Q1"], mix["Q2"], mix["Q3"], mix["Q4"],
            slate["price"].min(),
            slate["price"].max(),
            slate["exp_revenue_per_impression"].mean(),
            df["exp_revenue_per_impression"].median(),
            slate["rating"].mean(),
            df.attrs["prior_rating"],
            slate["days_on_platform"].median(),
        )
    )

    # Repairs made at load time are surfaced here so the numbers above are never
    # the product of a silent fix.
    for note in df.attrs.get("load_notes", []):
        print("data note: %s" % note)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Rank marketplace listings for the default homepage feed."
    )
    p.add_argument("-i", "--input", default="marketplace_dataset.csv", help="path to the listings CSV")
    p.add_argument("-o", "--output", default="top20_homepage_feed.csv", help="where to write the ranked slate")
    p.add_argument("-n", "--top", type=int, default=20, help="how many listings to return")
    p.add_argument("--quality-gamma", type=float, default=None, help="override the quality exponent")
    p.add_argument("--exploration-share", type=float, default=None, help="override the exploration slot share")
    p.add_argument("--price-cap-share", type=float, default=None, help="override the max share of the feed per price quartile")
    args = p.parse_args(argv)

    cfg = Config()
    if args.quality_gamma is not None:
        cfg.quality_gamma = args.quality_gamma
    if args.exploration_share is not None:
        cfg.exploration_slot_share = args.exploration_share
    if args.price_cap_share is not None:
        cfg.max_share_per_price_quartile = args.price_cap_share

    try:
        listings = load_listings(args.input, cfg)
    except FileNotFoundError:
        print("error: could not find %s" % args.input, file=sys.stderr)
        return 1
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    if len(listings) < 4:
        print("error: need at least 4 usable listings to rank", file=sys.stderr)
        return 1

    scored = score_listings(listings, cfg)
    slate = build_slate(scored, cfg, top_n=args.top)

    print_slate(slate, scored, cfg)

    # The ranking is already printed above, so a locked output file (one open in
    # Excel, for example) should not lose the run.
    try:
        # %.6g keeps the file readable and stable across platforms. Full
        # float64 precision here is noise rather than information.
        slate[DISPLAY].to_csv(args.output, index=False, float_format="%.6g")
    except OSError as exc:
        print("\nwarning: could not write %s (%s)" % (args.output, exc), file=sys.stderr)
        return 1
    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
