# Default Homepage Feed: Ranking 2,000 Marketplace Listings

`rank_feed.py` reads `marketplace_dataset.csv`, prints the top 20 listings to show a user and writes `top20_homepage_feed.csv`.

```bash
pip install pandas numpy
python rank_feed.py                 # print + write the top 20
```

## 1. The objective

A homepage has a fixed number of slots. Showing a listing costs one impression, and
impressions are the scarce resource. The question for each listing is not "how much has this
sold" but "what do we get back if we spend one impression on it".

That rules out the obvious rankers straight away. Sorting by total purchases mostly rewards
listings that have been around longer. Sorting by conversion rate ignores what the sale is
worth. The unit that matters is expected revenue per impression, adjusted for the fact that
not all revenue is equally good for the platform.

The formula:

```
score = price * cvr * quality_multiplier * freshness_multiplier
```

`price * cvr` is the expected revenue from showing the listing once. The quality
multiplier discounts revenue from listings people rate badly. The freshness multiplier is a
small boost for new listings.

The score has units, which is the main reason I chose this shape over a weighted sum of
normalised features. A score of 3.20 is a specific claim: this listing should return about
$3.20 of quality-adjusted revenue per homepage impression. That makes the ranking auditable,
and it means the cost of every product constraint can be quoted in the same currency.

The terms multiply rather than add so that a bad rating discounts revenue proportionally. A
2-star listing earning $5 an impression and a 2-star listing earning $1 an impression should
take the same percentage hit. An additive penalty would wipe out the small one and barely
touch the large one.

## 2. What is in the data

2,000 listings, no missing values, no duplicate IDs.

| Column | Min | 25% | Median | 75% | Max |
| --- | --- | --- | --- | --- | --- |
| `price` | $4.99 | $14.57 | $25.22 | $43.99 | $370.24 |
| `days_on_platform` | 1 | 233 | 504 | 761 | 999 |
| `historical_views` | 0 | 677 | 1,813 | 4,284 | 159,787 |
| `historical_purchases` | 0 | 24 | 79 | 221 | 16,000 |
| `average_rating` | 0.0 | 3.5 | 4.1 | 4.6 | 5.0 |
| `review_count` | 0 | 1 | 6 | 18 | 1,512 |

Totals: 7,369,685 impressions, 514,460 purchases, $10,646,193 revenue.

### Two things in the data that will catch a naive ranker

**`average_rating = 0.0` is a sentinel, not a score.** 293 listings show 0.0, and every one
of them has zero reviews. I checked directly: `(rating == 0) & (review_count > 0)` returns no
rows. The brief defines the scale as 1.0 to 5.0, and the lowest rating among reviewed
listings is 1.8. So 0.0 means "not rated yet". Taking it at face value would rank every new
listing below the worst product on the platform, which is backwards for a marketplace that
needs new supply.

**`ITEM_0913`.** The most-purchased listing on the platform: 16,000 sales, 32% conversion,
$4.79 revenue per impression. That is 3.5 times the catalogue average of $1.37 and the third
highest of all 2,000. Its rating is 2.1 stars across 1,200 reviews. Any revenue-only ranker
puts it on the front page.

There are also 2 listings with zero impressions, 29 with zero purchases, and `ITEM_0284`,
which has one impression and one purchase for a "100% conversion rate" that means nothing.

## 3. Three findings

### Finding 1: price and conversion cancel out

I split the catalogue into 20 price ventiles and pooled the traffic in each, using total
purchases over total views so that one tiny listing cannot swing a bucket.

| Ventile average price | Pooled conversion | price x conversion |
| --- | --- | --- |
| $5.46 | 20.5% | $1.12 |
| $9.80 | 14.1% | $1.38 |
| $17.36 | 6.7% | $1.17 |
| $30.08 | 4.3% | $1.30 |
| $55.84 | 2.6% | $1.45 |
| $87.21 | 1.7% | $1.48 |
| $147.16 | 1.2% | $1.82 |

Conversion falls 17-fold across the price ladder, but revenue per impression barely moves.
It ranges from $1.12 to $2.09 with a mean of $1.43. Regressing `log(cvr)` on `log(price)`
weighted by traffic gives:

```
log(cvr) = 0.0111 - 0.8953 * log(price)          weighted R^2 = 0.963
```

An elasticity of -0.895, close to -1. An elasticity of exactly -1 would mean price times
conversion is constant.

So neither price nor conversion rate ranks anything on its own. Rank by conversion and you
get a homepage of $4.99 items; rank by price and you get $300 items; both earn roughly the
same. The signal is in the residual, in listings that beat the baseline for their own price
point.

This is also why I could simplify. Since -0.895 is within noise of -1, I dropped the fitted
curve and let the two cancel exactly. On held-out traffic the fitted curve scored $2.810 per
impression against $2.811 for the simpler form.

### Finding 2: below roughly 300 impressions, conversion is mostly noise

If the spread between listings were pure sampling noise it would shrink as listings
accumulate traffic. If it is real quality it will not. So I measured the standard deviation
of revenue per impression by view octile:

| Median views | 149 | 477 | 888 | 1,467 | 2,211 | 3,338 | 5,341 | 11,502 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Std dev | 1.74 | 0.74 | 0.67 | 0.67 | 0.70 | 0.71 | 0.67 | 0.82 |

It collapses from 1.74 to about 0.70 and then stops. Below a few hundred impressions the
noise dominates, so shrinkage is necessary. Above that point the remaining spread of about
0.70 is real quality difference, so ranking is meaningful. The flat tail is what tells me the
amount of shrinkage can be measured rather than guessed.

### Finding 3: age is exposure, not quality

| Age decile | 1 (newest) | 3 | 5 | 7 | 10 (oldest) |
| --- | --- | --- | --- | --- | --- |
| Median impressions | 148 | 937 | 1,904 | 2,933 | 3,946 |
| Pooled conversion | 6.3% | 8.1% | 6.8% | 6.4% | 8.4% |

Impressions rise 27-fold with age while conversion stays between 6% and 8% with no trend.
Time on platform tells us how much a listing has been shown, not how good it is. A ranker
built purely on accumulated history is therefore a rich-get-richer loop with nothing in the
data to justify it, and any freshness term is a business subsidy rather than a relevance
correction. I kept mine small and labelled it as such.

## 4. Building the formula

### Term 1: `price * cvr`, expected revenue per impression

The raw conversion rate is a maximum-likelihood estimate on samples ranging from 0 to 159,787
impressions, so `ITEM_0284`'s 100% is worthless. The fix is empirical Bayes: shrink each
listing toward a prior, in proportion to how little data it has.

**Step 1, the prior.** From Finding 1, a listing's expected conversion before we see any of
its own data depends on its price:

```
R = total revenue / total impressions = $10,646,193 / 7,369,685 = $1.444593

prior conversion  p = R / price
```

A $220 listing is expected to convert at 1.4446/220, or 0.657%. A $5 listing is expected to
convert at 28.9%. One number, checkable by hand, in place of a fitted curve.

**Step 2, the Beta-Binomial.** Purchases out of impressions is binomial, and its conjugate
prior is Beta(alpha, beta), so the posterior is Beta(alpha + purchases, beta + views -
purchases) with mean:

```
cvr = (purchases + alpha) / (views + alpha + beta)
```

I anchor the prior by setting `alpha = alpha0`, a fixed pseudo-purchase count, with mean `p`.
That forces `beta = alpha0 * (1 - p) / p`, so `alpha + beta = alpha0 / p`. Substituting:

```
cvr = (purchases + alpha0) / (views + alpha0/p)
```

Fixing a pseudo-purchase count rather than a pseudo-view count is the part worth
understanding. The prior's weight in impressions is `alpha0 / p`, so it is worth 14
impressions for the cheapest listing and 1,051 for the dearest. That is the right behaviour,
because measuring a 0.6% rate as precisely as a 29% rate genuinely takes far more traffic. A
single "needs 100 views" threshold would be too strict at the cheap end and too lax at the
expensive end, and on held-out traffic that mistake costs 28% of realised revenue,
worse than using no shrinkage at all.

**Step 3, deriving alpha0 = 4.10.** Within each price ventile the observed spread splits in
two:

```
Var(observed cvr)  =  Var(real differences)  +  E[ p(1-p)/n ]
     measurable            what I want           binomial noise, measurable
```

For a Beta with strength `s = alpha + beta`, `Var = p(1-p)/(s+1)`, so `s = p(1-p)/Var_real - 1`
and `alpha = p * s`. Worked through for the top ventile:

| | value |
| --- | --- |
| total variance | 0.000043 |
| minus binomial noise | 0.000010 |
| leaves real variance | 0.000033 |
| strength `s` | 313.0 |
| `alpha = p * s` | 0.0104 x 313.0 = 3.27 |

Across all 20 ventiles: 4.15, 4.10, 4.94, 4.32, 4.74, 3.86, 5.20, 5.59, 4.10, 0.29, 4.40,
5.66, 3.98, 4.63, 3.55, 3.00, 3.51, 2.99, 3.68, 3.27.

The median is 4.10. I used the median rather than the mean because ventile 9 returned 0.29,
its total variance inflated tenfold by a single outlier. In fairness the mean would have been
4.00, so it made little difference here.

### Term 2: `quality_multiplier`

First a Bayesian average rating, so that one glowing review does not outrank 200:

```
rating_adj = (rating * reviews + 4.1752 * 10) / (reviews + 10)
```

`4.1752` is the mean rating of the 1,707 listings that have at least one review. The 293
unrated listings land on it exactly, so they are neither rewarded nor punished for having no
reviews yet.

Then the multiplier:

```
quality_multiplier = (rating_adj / 4.1752) ** gamma,     gamma = 2
```

This term is deliberately not fitted, and I want to be explicit about why. In this snapshot
rating does not predict conversion at all:

| Rating band | 3.0 and below | 3.0 to 3.5 | 3.5 to 4.0 | 4.0 to 4.5 | 4.5 to 5.0 |
| --- | --- | --- | --- | --- | --- |
| n | 45 | 200 | 401 | 549 | 512 |
| Revenue per impression | $1.41 | $1.42 | $1.42 | $1.43 | $1.42 |

The correlation between rating and conversion is -0.04. Flat. A ranker that only maximises
revenue would therefore ignore ratings, and put `ITEM_0913` on the front page.

It should not, because the cost of a bad purchase does not appear in this file. Refunds,
support load, churn and the customer who never comes back are all real and all absent from
the snapshot. So `gamma = 2` encodes a business prior about long-term value rather than a
statistical finding. In production I would calibrate it against a retention holdout. What I
can defend from this data is what it costs. On held-out traffic it earned +$0.015 per
impression more than `gamma = 0` (95% CI +0.007 to +0.022), so the answer is nothing.

There is also a hard floor. Any listing whose shrunk rating falls below 3.0 is excluded
regardless of revenue. It is applied to the shrunk rating so that one angry review cannot
delist a good product. Three listings hit it, `ITEM_0913` among them. Even at `gamma = 0` the
floor still catches it.

### Term 3: `freshness_multiplier`

```
freshness_multiplier = 1 + 0.15 * exp(-days / 30)
```

That is +15% on day 0, +5% at 33 days, and negligible past 90. Finding 3 showed age carries
no quality signal, so this cannot be justified as relevance. It is a bounded subsidy for new
supply, which is why it is small. The heavier lifting for new listings is done by the
exploration budget: 3 of the 20 slots are reserved for listings that have received fewer
impressions than their own prior is worth, meaning `views < alpha0/p`. 83 listings, or 4%,
qualify. Reserving slots rather than adding a score bonus caps the cost up front and cannot
be gamed by inflating a feature. A further rule caps any one catalogue price quartile at 8 of
the 20 slots, and ties break on `item_id` so the feed does not reshuffle between page loads.

## 5. Every constant and where it came from

The distinction that matters most in this submission is which numbers were measured and which
were chosen.

Measured from the data:

| Value | Derivation |
| --- | --- |
| `R = $1.444593` | Total revenue over total impressions, $10,646,193 / 7,369,685 |
| `alpha0 = 4.10` | Median of 20 per-ventile method-of-moments estimates (Section 4) |
| `4.1752` | Mean `average_rating` of the 1,707 listings with at least one review |
| `-0.895`, `R^2 0.963` | Traffic-weighted log-log regression across 20 price ventiles. This is the evidence for using `R/price`; it is not used at runtime |

Chosen by judgement:

| Value | Where | Reasoning |
| --- | --- | --- |
| `gamma = 2` | quality exponent | The revenue-versus-trust dial. Costs nothing held out and lifts slate rating from 4.23 to 4.59. Should be calibrated on retention |
| `10` | rating pseudo-reviews | Median listing has 6 reviews, 75th percentile has 18, so 10 puts a typical listing about halfway between its own rating and the platform mean |
| `3.0` | hard floor | About 1.2 stars below the platform mean. A backstop, not a fine gradation |
| `0.15`, `tau = 30d` | freshness | Kept small and visibly bounded, since age has no quality signal |
| `15%`, 3 slots | exploration | A slot budget caps the cost up front; a score bonus would not |
| `40%`, 8 slots | price diversity cap | Nearly free, for the reason in Section 7 |

All six sit in one `Config` block at the top of the script, so these arguments can happen in a
product review rather than inside the model.

## 6. Worked examples

`ITEM_1265`, rank 1. $220.00, 15,000 impressions, 450 purchases, 4.6 stars from 12 reviews,
400 days old.

```
prior p      = 1.444593 / 220.00            = 0.006566   0.66% expected at this price
prior weight = 4.10 / 0.006566              = 624 impressions
cvr          = (450 + 4.10) / (15000 + 624) = 0.029064   raw was 0.030000, barely moved
revenue/imp  = 220.00 * 0.029064            = $6.3940    4.4x the $1.44 baseline
rating_adj   = (4.6*12 + 4.1752*10)/(12+10)  = 4.4069     12 reviews, pulled toward 4.18
quality      = (4.4069 / 4.1752)**2          = 1.1141
freshness    = 1 + 0.15 * exp(-400/30)       = 1.0000     400 days old, no boost
score        = 6.3940 * 1.1141 * 1.0000      = 7.1234
```

15,000 impressions against a 624-impression prior, so its own data dominates. It ranks first
because it earns 4.4 times what a $220 listing normally earns.

`ITEM_0913`, rank 1103, excluded. $14.99, 50,000 impressions, 16,000 purchases, 2.1 stars
from 1,200 reviews.

```
cvr          = (16000 + 4.10) / (50000 + 42.5) = 0.319810   32%, essentially unshrunk
revenue/imp  = 14.99 * 0.319810                 = $4.7940    3.5x the catalogue norm
rating_adj   = (2.1*1200 + 4.1752*10)/1210      = 2.1172     1,200 reviews, the prior cannot save it
quality      = (2.1172 / 4.1752)**2             = 0.2571     a 74% cut
score        = 4.7940 * 0.2571                  = 1.2327     1103rd
eligible?    no, 2.1172 is below the 3.0 floor
```

`ITEM_0284`, rank 378. $25.00, one impression, one purchase, unrated, 5 days old.

```
prior weight = 4.10 / (1.444593/25) = 71 impressions
cvr          = (1 + 4.10) / (1 + 71)  = 0.0709    the 100% becomes 7.1%
```

One impression against a 71-impression prior, so the prior wins 98 to 2. Shrinkage on its own
moves this listing from apparent best on the platform to 378th.

`ITEM_0835`, slate rank 20 via an exploration slot. $14.20, 32 impressions, 5 purchases, 9
days old.

```
cvr          = (5 + 4.10) / (32 + 40.3)  = 0.125861   raw 0.156250, shrunk toward 10.2%
freshness    = 1 + 0.15 * exp(-9/30)     = 1.1111
score        = 2.0402                               354th overall
```

354th on score, but it reaches the feed through one of the three reserved exploration
slots.

## 7. Tradeoffs

| Decision | Cost | Benefit |
| --- | --- | --- |
| `gamma = 2` instead of 0 | None measurable, +$0.015 held out with the CI excluding zero | Slate rating 4.23 to 4.59, and the 2.1-star bestseller stays off the homepage |
| 3 of 20 slots for exploration | Part of the 1.3% below | New and neglected supply gets a guaranteed path to first impressions |
| At most 8 of 20 per price quartile | 1.3% combined | A shelf spanning $14 to $220 rather than one clustered at the top |
| Posterior mean rather than a lower bound | Some exposure to the winner's curse | The correct estimator when expected revenue is the objective |
| No personalisation | Ignores any signal a session might carry | The brief's user has not searched, so there is no signal yet |

Both product constraints together cost 1.3% of expected revenue per impression. They are
cheap for the same structural reason: revenue per impression is nearly flat across the price
ladder, so there are always well-rated alternatives earning near the top of the range.

## 8. Assumptions

1. `average_rating = 0.0` means unrated rather than zero-star. This is the highest-impact
   assumption in the submission. If it is wrong, the treatment of all 293 new listings
   inverts.
2. All 2,000 listings are in stock and interchangeable for a slot. No category, seller,
   inventory or geography column exists to say otherwise.
3. An impression is the scarce resource, so the objective is value per impression rather than
   total historical value.
4. `price` is gross revenue and the take rate is uniform. With a category-varying commission
   the objective becomes contribution margin, which is a one-line change.
5. The counts are stationary. With no timestamps, a purchase from 900 days ago is weighted
   like one from yesterday.
6. All users are alike, because a user who has not searched has given us no signal. This is
   the cold-start prior that personalisation should replace.


## 9. Why this helps the platform

On the demand side, the default feed is the platform's first impression and an implicit
endorsement of what it shows. Ranking on revenue per impression rather than historical totals
makes the homepage sell rather than merely display, and the quality terms keep what it sells
from costing us the customer, at no measurable revenue cost.

On the supply side, sellers stay where they get distribution. Shrinking new listings toward a
fair price-conditional prior rather than toward zero, plus a guaranteed exploration budget,
gives a good new listing a measurable path onto the homepage instead of needing traffic to
earn traffic.

Operationally, every business judgement sits in one `Config` block with its cost quantified in
dollars per impression, so arguments about `gamma` or the exploration share happen in a
product review rather than inside the model.

What I would do next, in order of value: log timestamps and decay old activity; add category
so real diversity becomes possible; replace the fixed exploration budget with Thompson
sampling from the Beta posterior the script already computes, which is about ten lines and
self-tuning; calibrate `gamma` against a retention holdout; then personalise, leaving this as
the cold-start fallback.
