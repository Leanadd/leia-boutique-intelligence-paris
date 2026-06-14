import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

base = "/Users/leana/Downloads/Mes documents/leia_paris/knowledge_base/"

purchases = pd.read_csv(base + "purchase_history_3500.csv")
purchases["date"] = pd.to_datetime(purchases["date"], errors="coerce")

print(f"Purchases loaded : {purchases.shape}")
print(f"Date range : {purchases['date'].min()} → {purchases['date'].max()}")
print(f"Collections : {purchases['collection'].unique()}")

# ── PREPARE TIME SERIES ───────────────────────────────────────────────────────

# Prophet requires exactly two columns :
# "ds" = datestamp (datetime)
# "y"  = value to forecast
#
# we aggregate purchases by month
# monthly granularity smooths day-to-day noise
# and reveals meaningful seasonal patterns

monthly = (
    purchases
    .groupby(purchases["date"].dt.to_period("M"))
    .size()
    .reset_index()
)
monthly.columns = ["ds", "y"]
monthly["ds"] = monthly["ds"].dt.to_timestamp()

print(f"\nMonthly aggregation : {monthly.shape}")
print(monthly.head(8))

# visualize raw time series before modeling
plt.figure(figsize=(12, 5))
plt.plot(monthly["ds"], monthly["y"],
         marker="o", color="#534AB7", linewidth=2)
plt.title("Monthly Purchase Volume — LÉIA Paris (2022–2025)")
plt.xlabel("Month")
plt.ylabel("Number of purchases")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("monthly_purchases.png", dpi=150)
plt.show()
print("✅ Graph saved : monthly_purchases.png")

# filter to complete years only
# 2025 data is incomplete (generated up to March 2025 only)
# including it would bias Prophet toward a false downward trend

monthly = monthly[monthly["ds"] < "2025-01-01"].copy()

print(f"After filtering : {monthly.shape}")
print(f"Period : {monthly['ds'].min()} → {monthly['ds'].max()}")

# ── BLOC 2 : PROPHET MODEL ────────────────────────────────────────────────────

# filter to complete years only — 2020 to 2024
monthly = monthly[monthly["ds"] < "2025-01-01"].copy()
monthly = monthly[monthly["ds"] >= "2020-01-01"].copy()

print(f"Period : {monthly['ds'].min()} → {monthly['ds'].max()}")
print(f"Data points : {len(monthly)}")

# ── ADD LUXURY EVENTS AS HOLIDAYS ─────────────────────────────────────────────

# Prophet supports external events via "holidays" dataframe
# this tells the model : around these dates, expect abnormal activity
# lower_window = days before the event to capture pre-event buying
# upper_window = days after the event

holidays = pd.DataFrame({
    "holiday": [
        # Paris Fashion Week — March and September
        "Paris Fashion Week", "Paris Fashion Week",
        "Paris Fashion Week", "Paris Fashion Week",
        "Paris Fashion Week", "Paris Fashion Week",
        "Paris Fashion Week", "Paris Fashion Week",
        "Paris Fashion Week", "Paris Fashion Week",
        # Art Basel Paris — October
        "Art Basel Paris", "Art Basel Paris",
        "Art Basel Paris", "Art Basel Paris",
        "Art Basel Paris",
        # Chinese New Year — Jan/Feb
        "Chinese New Year", "Chinese New Year",
        "Chinese New Year", "Chinese New Year",
        "Chinese New Year",
        # Eid Al-Fitr — variable month
        "Eid Al-Fitr", "Eid Al-Fitr",
        "Eid Al-Fitr", "Eid Al-Fitr",
        "Eid Al-Fitr",
        # December holidays
        "December Holidays", "December Holidays",
        "December Holidays", "December Holidays",
        "December Holidays",
    ],
    "ds": pd.to_datetime([
        # Fashion Week March
        "2020-03-02","2021-03-01","2022-03-07","2023-03-06","2024-03-04",
        # Fashion Week September
        "2020-09-28","2021-09-27","2022-09-26","2023-09-25","2024-09-23",
        # Art Basel Paris October
        "2020-10-21","2021-10-21","2022-10-20","2023-10-20","2024-10-18",
        # Chinese New Year
        "2020-01-25","2021-02-12","2022-02-01","2023-01-22","2024-02-10",
        # Eid Al-Fitr
        "2020-05-24","2021-05-13","2022-05-02","2023-04-21","2024-04-10",
        # December Holidays
        "2020-12-15","2021-12-15","2022-12-15","2023-12-15","2024-12-15",
    ]),
    "lower_window": -3,   # 3 days before event
    "upper_window": 3,    #  days after event
})

# ── TRAIN PROPHET ─────────────────────────────────────────────────────────────

# Prophet decomposes the time series into 3 components :
# trend     → overall direction (growing? declining?)
# seasonality → recurring patterns (weekly, monthly, yearly)
# holidays  → impact of specific events
#
# yearly_seasonality=True : learns annual patterns
# weekly_seasonality=False : monthly data, no weekly signal
# changepoint_prior_scale : flexibility of the trend
#   low value (0.05) = smooth trend, avoids overfitting
#   high value (0.5) = flexible trend, follows data closely

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    holidays=holidays,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=2,    
    holidays_prior_scale=2,       # limit the impact of holidays on the forecast
)

model.fit(monthly)
print("✅ Prophet model trained")

# ── FORECAST 12 MONTHS AHEAD ──────────────────────────────────────────────────

# make_future_dataframe creates a dataframe with future dates
# periods=12 → forecast 12 months into the future (2025)
# freq="MS"  → Month Start frequency

future = model.make_future_dataframe(periods=12, freq="MS")
forecast = model.predict(future)

print(f"\nForecast period : {forecast['ds'].iloc[-12]} → {forecast['ds'].iloc[-1]}")
print("\nPredicted monthly purchases for 2025 :")
future_only = forecast[forecast["ds"] >= "2025-01-01"][["ds","yhat","yhat_lower","yhat_upper"]]
future_only.columns = ["Month","Predicted","Lower bound","Upper bound"]
future_only["Predicted"] = future_only["Predicted"].round(0).astype(int)
future_only["Lower bound"] = future_only["Lower bound"].round(0).astype(int)
future_only["Upper bound"] = future_only["Upper bound"].round(0).astype(int)
print(future_only.to_string(index=False))

# ── VISUALIZE FORECAST ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1 : full forecast with confidence interval
axes[0].fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.2, color="#534AB7", label="Confidence interval"
)
axes[0].plot(
    forecast["ds"], forecast["yhat"],
    color="#534AB7", linewidth=2, label="Predicted"
)
axes[0].plot(
    monthly["ds"], monthly["y"],
    color="#1D9E75", linewidth=2,
    marker="o", markersize=4, label="Actual"
)
axes[0].axvline(
    pd.to_datetime("2025-01-01"),
    color="#D85A30", linestyle="--",
    linewidth=1.5, label="Forecast start"
)
axes[0].set_title("LÉIA Paris — Purchase Volume Forecast 2020–2025")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Number of purchases")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2 : 2025 forecast only — bar chart
future_2025 = forecast[forecast["ds"] >= "2025-01-01"].copy()
months_labels = [d.strftime("%b") for d in future_2025["ds"]]
colors_bars = ["#D85A30" if m in ["Mar","Oct","Feb","Dec"]
               else "#534AB7" for m in months_labels]

axes[1].bar(
    months_labels, future_2025["yhat"],
    color=colors_bars, edgecolor="white", alpha=0.85
)
axes[1].errorbar(
    months_labels, future_2025["yhat"],
    yerr=[
        future_2025["yhat"] - future_2025["yhat_lower"],
        future_2025["yhat_upper"] - future_2025["yhat"]
    ],
    fmt="none", color="black", capsize=4, linewidth=1
)
axes[1].set_title("2025 Monthly Forecast — LÉIA Paris (orange = key luxury events)")
axes[1].set_ylabel("Predicted purchases")
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("forecast_2025.png", dpi=150)
plt.show()
print("✅ Graph saved : forecast_2025.png")

# ── COLLECTION-LEVEL FORECAST ─────────────────────────────────────────────────

# run Prophet separately for each collection
# to identify which collections peak at which events

print("\n--- Collection-level trends ---")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Collection Trends & Forecast 2025 — LÉIA Paris", fontsize=14)

collection_colors = {
    "Eclipse": "#534AB7",
    "Hatching": "#A855F7",
    "Amazon": "#1D9E75",
    "Vanta": "#64748B"
}

for idx, collection in enumerate(["Eclipse", "Hatching", "Amazon", "Vanta"]):
    ax = axes2[idx // 2][idx % 2]

    # filter purchases by collection
    col_purchases = purchases[purchases["collection"] == collection]
    col_monthly = (
        col_purchases
        .groupby(col_purchases["date"].dt.to_period("M"))
        .size()
        .reset_index()
    )
    col_monthly.columns = ["ds", "y"]
    col_monthly["ds"] = col_monthly["ds"].dt.to_timestamp()
    col_monthly = col_monthly[col_monthly["ds"] < "2025-01-01"].copy()

    # train Prophet per collection
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=2,
        holidays_prior_scale=2,
    )
    m.fit(col_monthly)

    future_col = m.make_future_dataframe(periods=12, freq="MS")
    forecast_col = m.predict(future_col)

    color = collection_colors[collection]

    ax.fill_between(
        forecast_col["ds"],
        forecast_col["yhat_lower"],
        forecast_col["yhat_upper"],
        alpha=0.15, color=color
    )
    ax.plot(forecast_col["ds"], forecast_col["yhat"],
            color=color, linewidth=2, label="Forecast")
    ax.plot(col_monthly["ds"], col_monthly["y"],
            color=color, linewidth=1.5,
            marker="o", markersize=3, alpha=0.7, label="Actual")
    ax.axvline(pd.to_datetime("2025-01-01"),
               color="#D85A30", linestyle="--", linewidth=1)
    ax.set_title(f"{collection} Collection")
    ax.set_ylabel("Purchases")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("forecast_collections.png", dpi=150)
plt.show()
print("✅ Graph saved : forecast_collections.png")

# ── PROPHET RESULTS INTERPRETATION ───────────────────────────────────────────

# OVERALL FORECAST (2025)
# Prophet identifies a clear growth trend from 2020 to 2024
# reflecting the boutique's development narrative :
# 2020 : COVID impact — soft opening, low volume
# 2021 : recovery — client base building
# 2022 : acceleration — brand awareness growing
# 2023 : peak year — full operational maturity
# 2024 : stable high — consistent performance
#
# 2025 forecast reflects this maturity :
# stable monthly volume between 78 and 143 purchases
# with clear seasonal peaks aligned with luxury events :
# → March  (+43 vs avg) : Paris Fashion Week
# → February (+10)      : Chinese New Year
# → October (+21)       : Art Basel Paris
# → December (+18)      : Holiday gifting season
# → August  (-22)       : Paris summer exodus (trough)

# COLLECTION-LEVEL FORECAST
# all four collections show similar forecast shapes 
# this is expected given that the synthetic dataset was generated with uniform temporal rules across collections
#
# in a real-world dataset, each collection would exhibit distinct seasonal drivers

# KEY BUSINESS OUTPUT
# this forecast enables LÉIA's teams to :
# → plan advisor staffing around peak months (Mar, Oct, Dec)
# → time collection launches before seasonal peaks
# → prepare inventory and after-sales capacity in advance
# → identify slow months (Aug, Nov, Jan) for training and maintenance