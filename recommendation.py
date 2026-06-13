import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

base = "/Users/leana/Downloads/Mes documents/leia_paris/knowledge_base/"

clients = pd.read_csv(base + "clients_segmented.csv")
purchases = pd.read_csv(base + "purchase_history_3500.csv")
products = pd.read_csv(base + "leia_products.csv")

print(f"Clients  : {clients.shape}")
print(f"Purchases: {purchases.shape}")
print(f"Products : {products.shape}")

# ── BUILD TRAINING DATASET ────────────────────────────────────────────────────

# the goal : predict which product a client will buy next
# based on their profile (age, spend, collection affinity, segment...)
# 
# each row = one purchase
# features X = client profile at time of purchase
# label y = which product they bought
#
# we merge purchases with client profiles
# so each transaction carries the full client context

df = purchases.merge(
    clients[[
        "client_id", "age", "total_spent_usd", "purchase_count",
        "vip_tier", "preferred_collections", "budget_range",
        "cluster", "segment"
    ]],
    on="client_id",
    how="left"
)

print(f"\nMerged dataset : {df.shape}")
print(f"Columns : {list(df.columns)}")

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

# convert categorical variables to numbers
# LabelEncoder assigns a unique integer to each category

le_tier = LabelEncoder()
le_collection = LabelEncoder()
le_occasion = LabelEncoder()
le_segment = LabelEncoder()

df["vip_tier_enc"]              = le_tier.fit_transform(df["vip_tier"])
df["preferred_collections_enc"] = le_collection.fit_transform(df["preferred_collections"])
df["occasion_enc"]              = le_occasion.fit_transform(df["occasion"])
df["segment_enc"]               = le_segment.fit_transform(df["segment"])

# extract budget max from budget_range string "5000-15000" → 15000
# the upper bound is more meaningful for predicting purchase behavior
df["budget_max"] = df["budget_range"].str.split("-").str[1].astype(int)

# define features (X) and target (y)
# X = what we know about the client
# y = what product they bought → this is what we want to predict

features = [
    "age",                       
    "total_spent_usd",           
    "purchase_count",            
    "vip_tier_enc",              
    "preferred_collections_enc", 
    "occasion_enc",              
    "budget_max",                
    "cluster",                   # which segment they belong to
    "price_usd",                 
]

X = df[features]
y = df["product_id"]   

print(f"Features shape : {X.shape}")
print(f"Unique products to predict : {y.nunique()}")
print(f"\nFeature preview :")
print(X.head(3))

# ── TRAIN / TEST SPLIT + MODEL TRAINING ───────────────────────────────────────

# split data into training set (80%) and test set (20%)
# we evaluate its performance on the test set 
# random_state=42 ensures reproducibility — same split every time

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set : {X_train.shape}")
print(f"Test set     : {X_test.shape}")

# ── RANDOM FOREST CLASSIFIER ──────────────────────────────────────────────────

# we start with Random Forest before XGBoost
# Random Forest builds many decision trees and combines their predictions
# it handles mixed features well (numerical + encoded categorical)
# class_weight="balanced" compensates for the imbalance between products
# some products are bought more often than others 
# without this, the model would over-predict popular products and ignore rare ones

model = RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    class_weight="balanced",
    random_state=42,
    n_jobs=-1              # use all CPU cores to speed up training
)

print("\nTraining model...")
model.fit(X_train, y_train)
print("✅ Model trained")

# ── EVALUATION ────────────────────────────────────────────────────────────────

# evaluate on the test set
# precision = when the model predicts product X, how often is it right?
# recall    = of all purchases of product X, how many did the model catch?
# f1-score  = harmonic mean of precision and recall

y_pred = model.predict(X_test)

print("\nModel performance on test set :")
print(classification_report(y_test, y_pred, zero_division=0))

#Accuracy 0.80 → The model predicts the correct product 8 times out of 10
# verage F1 score 0.82 → Balanced average performance across all products 

# ── RECOMMENDATION FUNCTION ───────────────────────────────────────────────────

# given a client profile, return the top 3 recommended products
# filtering out products the client already owns
# and products outside their budget range

def recommend_for_client(client_id, top_n=3):
    # retrieve client profile
    client = clients[clients["client_id"] == client_id]

    if client.empty:
        print(f"Client {client_id} not found.")
        return

    client = client.iloc[0]

    # products already owned by this client
    already_bought = purchases[
        purchases["client_id"] == client_id
    ]["product_id"].tolist()

    # build a candidate row for each product not yet owned
    budget_max = int(str(client["budget_range"]).split("-")[1])

    candidates = products[
        (~products["product_id"].isin(already_bought)) &
        (products["price_usd"] <= budget_max * 1.2)
    ].copy()

    if candidates.empty:
        print(f"No new products to recommend for {client_id}.")
        return

    # encode client features for each candidate product
    occasion_default = df["occasion_enc"].mode()[0]

    candidates["age"]                       = client["age"]
    candidates["total_spent_usd"]           = client["total_spent_usd"]
    candidates["purchase_count"]            = client["purchase_count"]
    candidates["vip_tier_enc"]              = le_tier.transform([client["vip_tier"]])[0]
    candidates["preferred_collections_enc"] = le_collection.transform(
        [client["preferred_collections"]]
    )[0]
    candidates["occasion_enc"]              = occasion_default
    candidates["budget_max"]                = budget_max
    candidates["cluster"]                   = client["cluster"]

    X_candidates = candidates[features]

    # predict probability for each product
    # predict_proba returns the probability of each class (product)
    # we take the probability of the correct product_id
    probas = model.predict_proba(X_candidates)
    classes = list(model.classes_)

    scores = []
    for idx, row in candidates.iterrows():
        pid = row["product_id"]
        if pid in classes:
            score = probas[candidates.index.get_loc(idx)][classes.index(pid)]
        else:
            score = 0
        scores.append(score)

    candidates["score"] = scores
    top3 = candidates.nlargest(top_n, "score")

    # display recommendations
    print(f"\n{'='*55}")
    print(f"Recommendations for {client['first_name']} {client['last_name']}")
    print(f"Segment : {client['segment']} | VIP : {client['vip_tier']} | Budget : ${budget_max:,}")
    print(f"Already owns : {len(already_bought)} pieces")
    print(f"{'='*55}")

    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"\n#{rank} {row['name']} ({row['collection']})")
        print(f"   Price    : ${row['price_usd']:,}")
        print(f"   Score    : {row['score']:.2%}")

# ── TEST THE FUNCTION ─────────────────────────────────────────────────────────

# test on a few clients from different segments
print("\n--- Testing recommendation function ---")

# pick one client from each segment
for segment in clients["segment"].unique():
    sample = clients[clients["segment"] == segment].iloc[0]
    recommend_for_client(sample["client_id"])

    # ── MODEL INTERPRETATION ──────────────────────────────────────────────────────

# overall accuracy : 80% on 42 products — strong result
# the model correctly identifies the purchased product 8 times out of 10

# WHAT WORKS WELL
# Vanta watches : highest F1 scores (VAN002, VAN004, VAN006 = 1.00)
#   → watch buyers have a very distinct profile (high budget, tech-minimalist)
#   → strong signal for the model

# Eclipse identity pieces : strong for segment-aligned clients
#   → Identity Seekers receive Eclipse recommendations with high confidence
#   → Feng Liu : 57% score on Unity Chain Necklace ✅
#   → Tomoko Yamaguchi (New Entrant) : 79% on Identity Medallion ✅

# WHAT IS MORE CHALLENGING
# ECL006, ECL008 : lowest F1 scores (0.38, 0.33)
#   → these Eclipse pieces are bought by very diverse profiles
#   → hard to find a clear signal — consistent with Eclipse's
#     "beyond definition" philosophy

# Diamond VIPs with 15+ purchases
#   → catalog nearly exhausted for these clients
#   → model recommends what's left, not necessarily what fits best
#   → real-world solution : flag these clients for bespoke service
#     rather than standard recommendations

# KEY BUSINESS INSIGHT
# The recommendation engine works best for clients
# with fewer than 10 purchases and a clear collection affinity
# For top VIP clients (Diamond tier), the model should trigger
# a "bespoke service" alert rather than a standard recommendation

# ── SAVE THE MODEL ────────────────────────────────────────────────────────────

import joblib

# save the trained model and encoders
# joblib is the standard way to persist sklearn models
# this allows loading the model later without retraining

joblib.dump(model, base + "recommendation_model.pkl")
joblib.dump(le_tier, base + "le_tier.pkl")
joblib.dump(le_collection, base + "le_collection.pkl")
joblib.dump(le_occasion, base + "le_occasion.pkl")
joblib.dump(le_segment, base + "le_segment.pkl")

print("\n✅ Model saved : recommendation_model.pkl")
print("✅ Encoders saved : le_tier, le_collection, le_occasion, le_segment")