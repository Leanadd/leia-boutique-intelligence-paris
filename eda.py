import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement
base = "/Users/leana/Downloads/Mes documents/leia_paris/knowledge_base/"

clients = pd.read_csv(base + "client_profiles_500.csv")
purchases = pd.read_csv(base + "purchase_history_3500.csv")

# Première vision
print("=== CLIENTS ===")
print(f"Shape : {clients.shape}")
print(clients.dtypes)
print("\n")
print(clients.describe())

print("\n=== PURCHASES ===")
print(f"Shape : {purchases.shape}")
print(purchases.describe()) 

# ── VISUALISATIONS ────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("LÉIA Paris — EDA", fontsize=16)

# 1. Distribution des âges
axes[0,0].hist(clients["age"], bins=20, color="#534AB7", edgecolor="white")
axes[0,0].set_title("Distribution des âges")
axes[0,0].set_xlabel("Âge")

# 2. VIP tiers
tier_counts = clients["vip_tier"].value_counts()
axes[0,1].bar(tier_counts.index, tier_counts.values, color=["#94a3b8","#fbbf24","#cbd5e1","#3b82f6"])
axes[0,1].set_title("VIP Tier Distribution")

# 3. Total spent distribution
axes[0,2].hist(clients["total_spent_usd"], bins=30, color="#1D9E75", edgecolor="white")
axes[0,2].set_title("Distribution total_spent_usd")
axes[0,2].set_xlabel("USD")

# 4. Collections préférées
col_counts = clients["preferred_collections"].str.split(",").explode().str.strip().value_counts()
axes[1,0].bar(col_counts.index, col_counts.values, color="#D85A30", edgecolor="white")
axes[1,0].set_title("Collections préférées")
axes[1,0].tick_params(axis='x', rotation=45)

# 5. Achats par mois
purchases["date"] = pd.to_datetime(purchases["date"], errors="coerce")
purchases["month"] = purchases["date"].dt.month
monthly = purchases.groupby("month").size()
axes[1,1].plot(monthly.index, monthly.values, marker="o", color="#534AB7", linewidth=2)
axes[1,1].set_title("Achats par mois")
axes[1,1].set_xlabel("Mois")

# 6. Prix par collection
purchases.boxplot(column="price_usd", by="collection", ax=axes[1,2])
axes[1,2].set_title("Prix par collection")
axes[1,2].set_xlabel("")
plt.suptitle("")

plt.tight_layout()
plt.savefig("eda_visualisations.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Graphiques sauvegardés dans eda_visualisations.png")