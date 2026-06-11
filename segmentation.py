import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load preprocessed data
base = "/Users/leana/Downloads/Mes documents/leia_paris/knowledge_base/"

X_scaled = pd.read_csv(base + "clients_ml_ready.csv")
clients = pd.read_csv(base + "clients_preprocessed.csv")

print(f"Data loaded : {X_scaled.shape}")

# ── FIND THE OPTIMAL NUMBER OF CLUSTERS : ELBOW METHOD ────

# K-Means requires us to specify the number of clusters upfront
# we don't know the right number yet so we test from 2 to 10
# for each k, we compute inertia :
# the sum of squared distances between each point and its cluster center
# as k increases, inertia decreases we look for the "elbow" the point where adding one more cluster stops improving much

inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"k={k} → inertia: {kmeans.inertia_:.0f}")

# plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker="o", color="#534AB7", linewidth=2)
plt.title("Elbow Method — Optimal Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_range)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_curve.png", dpi=150)
plt.show()
print("\n✅ Graph saved : elbow_curve.png")

# ── TRAIN K-MEANS WITH k=5 ─────────

# k=5 chosen based on elbow curve inflection point and business logic
# 5 segments make sense for LÉIA's client diversity :
# high-value collectors, young self-gifters, mid-range international, VIP top spenders, entry-level occasional buyers

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clients["cluster"] = kmeans.fit_predict(X_scaled)

# cluster size overview
print("Cluster distribution :")
print(clients["cluster"].value_counts().sort_index())

# ── VISUALIZE CLUSTERS WITH PCA ───────────────────────────────────────────────

# PCA reduces our 11 features down to 2 dimensions
# so we can plot clusters on a 2D chart
# it finds the 2 axes that capture the most variance in the data
# we lose some information but gain visual interpretability

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nVariance explained by PCA : {pca.explained_variance_ratio_.sum()*100:.1f}%")

# plot
colors = ["#534AB7", "#1D9E75", "#D85A30", "#185FA5", "#F5C518"]
labels = [f"Cluster {i}" for i in range(5)]

plt.figure(figsize=(10, 7))
for i in range(5):
    mask = clients["cluster"] == i
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colors[i], label=labels[i],
        alpha=0.6, s=50, edgecolors="white", linewidth=0.3
    )

plt.title("K-Means Clustering — LÉIA Paris Clients (PCA 2D)")
plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("clusters_pca.png", dpi=150)
plt.show()
print("✅ Graph saved : clusters_pca.png")

# No empty clusters, no clusters that override all the others

# ── PROFILE EACH CLUSTER ──────────────────────────────────────────────────────

# for each cluster, compute the average of key variables
# this tells us what kind of client each cluster represents

profile_cols = [
    "total_spent_usd", "purchase_count", "age",
    "col_Amazon", "col_Eclipse", "col_Hatching", "col_Vanta",
    "vip_Diamond", "vip_Gold", "vip_Member", "vip_Platinum"
]

profile = clients.groupby("cluster")[profile_cols].mean().round(2)

print("\nCluster profiles :")
print(profile.to_string())

# most common collection per cluster
print("\nMost common collection per cluster :")
for i in range(5):
    subset = clients[clients["cluster"] == i]
    top_col = subset["preferred_collections"].value_counts().index[0]
    top_tier = subset["vip_tier"].value_counts().index[0]
    avg_spent = subset["total_spent_usd"].mean()
    print(f"Cluster {i} → collection: {top_col} | tier: {top_tier} | avg spent: ${avg_spent:,.0f}")

    # ── BUSINESS INTERPRETATION ───────────────────────────────────────────────────

# K-Means identified 5 distinct client segments based on
# spending behavior, collection affinity, purchase frequency, and VIP tier
#
# Cluster 0 — "The Elegant Gold" (173 clients, 35%)
#   Core boutique clientele. Gold tier, balanced between Hatching and Amazon.
#   Regular buyers with mid-range spend ($57k avg).
#   Strategy : loyalty programs, early access to new collections,
#               personalized outreach on milestones
#
# Cluster 1 — "The Identity Seekers" (142 clients, 28%)
#   Eclipse-dominant segment, strongly aligned with LÉIA's inclusivity values.
#   Gold tier, lower spend but high brand attachment.
#   Strategy : community events, Eclipse new drops, storytelling-driven outreach
#
# Cluster 2 — "The Platinum Collectors" (116 clients, 23%)
#   Serious collectors. Platinum tier, high frequency (11 purchases avg).
#   Multi-collection buyers — Amazon + Hatching.
#   Strategy : private appointments, preview access, trade-in program
#
# Cluster 3 — "The New Entrants" (43 clients, 9%)
#   First-time or occasional buyers. Member tier, low spend ($9k avg).
#   Highest growth potential if converted to Gold.
#   Strategy : welcome journey, Chrysalis Room invitation,
#               entry-level pieces recommendation
#
# Cluster 4 — "The Diamond VIPs" (26 clients, 5%)
#   Top spenders. Diamond tier, $348k avg, 17 purchases avg.
#   Amazon + Vanta dominant — statement pieces and watches.
#   Strategy : dedicated advisor, home visits, exclusive previews,
#               bespoke pieces, highest priority for any new launch
#
# Key insight : age is nearly identical across all clusters (~43 years)
# → age is NOT a differentiating factor at LÉIA Paris
# → spending level and collection affinity are the true behavioral drivers

# ── ASSIGN BUSINESS LABELS ────────────────────────────────────────────────────

# give each cluster a meaningful business name
# based on their spending, collection affinity, and VIP tier
cluster_labels = {
    0: "The Elegant Gold",
    1: "The Identity Seekers",
    2: "The Platinum Collectors",
    3: "The New Entrants",
    4: "The Diamond VIPs"
}

clients["segment"] = clients["cluster"].map(cluster_labels)

# ── FINAL VISUALIZATION ───────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("LÉIA Paris — Client Segmentation", fontsize=14)

colors = ["#534AB7", "#1D9E75", "#D85A30", "#185FA5", "#F5C518"]

# 1. Segment sizes
segment_counts = clients["segment"].value_counts()
axes[0].barh(segment_counts.index, segment_counts.values,
             color=colors, edgecolor="white")
axes[0].set_title("Clients per Segment")
axes[0].set_xlabel("Number of clients")

# 2. Avg spend per segment
avg_spend = clients.groupby("segment")["total_spent_usd"].mean().sort_values()
axes[1].barh(avg_spend.index, avg_spend.values,
             color=colors, edgecolor="white")
axes[1].set_title("Avg Spend per Segment")
axes[1].set_xlabel("USD")
axes[1].xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
)

# 3. Avg purchase count per segment
avg_count = clients.groupby("segment")["purchase_count"].mean().sort_values()
axes[2].barh(avg_count.index, avg_count.values,
             color=colors, edgecolor="white")
axes[2].set_title("Avg Purchase Count per Segment")
axes[2].set_xlabel("Number of purchases")

plt.tight_layout()
plt.savefig("segments_profile.png", dpi=150)
plt.show()

# save final dataset with segments
clients.to_csv(base + "clients_segmented.csv", index=False)
print("✅ Saved : clients_segmented.csv")
print("✅ Saved : segments_profile.png")

# ── SEGMENTATION RESULTS INTERPRETATION ──────────────────────────────────────

# Visual analysis of the 3 charts reveals a classic luxury retail pattern :

# CLIENTS PER SEGMENT
# The Elegant Gold dominates in volume (173 clients, 35%)
# The Diamond VIPs are the rarest group (26 clients, 5%)
# → typical luxury pyramid : large base, very small top
# → most clients are mid-tier, which is healthy for sustainable revenue

# AVG SPEND PER SEGMENT
# Diamond VIPs spend $348k on average 6x more than Elegant Gold ($57k)
# New Entrants are nearly invisible at $9k
# → the top 5% of clients likely generate 30-40% of total revenue
# → classic Pareto principle applied to luxury retail

# AVG PURCHASE COUNT PER SEGMENT
# Diamond VIPs average 17 purchases, they come back consistently
# New Entrants average 1-2 purchases; retention is the key challenge here
# → loyalty is the real growth lever, not acquisition

# KEY BUSINESS INSIGHT
# Clusters 2 + 4 (Platinum Collectors + Diamond VIPs) represent 28% of clients
# but likely account for 60-70% of total boutique revenue
# → priority should be protecting and nurturing these two segments first
# → converting New Entrants (Cluster 3) to Elegant Gold (Cluster 0)
#   is the highest-ROI growth opportunity

# NOTE ON AGE
# Age is nearly identical across all clusters (~43 years)
# → age is NOT a differentiating behavioral factor at LÉIA Paris
# → spending level and collection affinity are the true identity drivers
# → this aligns with LÉIA's brand philosophy : identity over demographics