import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

# Chargement
base = "/Users/leana/Downloads/Mes documents/leia_paris/knowledge_base/"

clients = pd.read_csv(base + "client_profiles_500.csv")
purchases = pd.read_csv(base + "purchase_history_3500.csv")

print(f"Clients  : {clients.shape}")
print(f"Purchases: {purchases.shape}")

# ── BLOC 2 : TRAITEMENT OUTLIER + NORMALISATION ────

# LOG-TRANSFORMATION
# total_spent_usd est très dispersé (min 2k, max 600k)
# sans transformation, le client à 600k domine tout
# log1p compresse les grandes valeurs tout en gardant les proportions
clients["total_spent_log"] = np.log1p(clients["total_spent_usd"])

# CAP DE L'OUTLIER
# on plafonne à la valeur du 99e percentile
# le client à 600k reste dans le dataset mais son influence sur le modèle est limitée 
# clip(upper=cap) remplace toute valeur au-dessus du cap par le cap
cap = clients["total_spent_usd"].quantile(0.99)
clients["total_spent_capped"] = clients["total_spent_usd"].clip(upper=cap)

print(f"Cap appliqué à : ${cap:,.0f}")
print(f"total_spent_usd    — min: {clients['total_spent_usd'].min():,} / max: {clients['total_spent_usd'].max():,}")
print(f"total_spent_log    — min: {clients['total_spent_log'].min():.2f} / max: {clients['total_spent_log'].max():.2f}")
print(f"total_spent_capped — max après cap: {clients['total_spent_capped'].max():,}")

# ── BLOC 3 : ENCODAGE DES VARIABLES CATÉGORIELLES ──────

# ONE-HOT ENCODING DES COLLECTIONS
# on transforme en colonnes binaires 0/1
# get_dummies crée une colonne par valeur unique
collections_dummies = clients["preferred_collections"].str.get_dummies(sep=", ")
collections_dummies.columns = ["col_" + c for c in collections_dummies.columns]
clients = pd.concat([clients, collections_dummies], axis=1)

# ONE-HOT ENCODING DU VIP TIER
# Member/Gold/Platinum/Diamond devient 4 colonnes binaires
# on utilise pd.get_dummies directement sur la colonne
vip_dummies = pd.get_dummies(clients["vip_tier"], prefix="vip")
clients = pd.concat([clients, vip_dummies], axis=1)

# ENCODAGE DE L'ÂGE EN TRANCHES
# l'âge brut (22-72) peut être bruité pour la segmentation
# on le découpe en 4 tranches significatives pour le luxe :jeune adulte / milieu de carrière / senior / retraité
clients["age_group"] = pd.cut(
    clients["age"],
    bins=[0, 30, 45, 60, 100],
    labels=["22-30", "31-45", "46-60", "60+"]
)
age_dummies = pd.get_dummies(clients["age_group"], prefix="age")
clients = pd.concat([clients, age_dummies], axis=1)

print("Colonnes après encodage :")
new_cols = [c for c in clients.columns if c.startswith(("col_", "vip_", "age_"))]
print(new_cols)
print(f"\nDataset final : {clients.shape}")

# ── BLOC 4 : STANDARDISATION ────

# les features ont des échelles très différentes
# K-Means calcule des distances et sans standardisation, total_spent va dominer juste parce que ses valeurs sont plus grandes
# StandardScaler ramène tout à la même échelle :
# moyenne = 0, écart-type = 1 pour chaque colonne

# on sélectionne uniquement les colonnes numériques utiles pour le ML
features = [
    "total_spent_log",    # dépenses normalisées
    "purchase_count",     # fréquence d'achat
    "age",                # âge
    "col_Amazon",         # collection préférée
    "col_Eclipse",
    "col_Hatching",
    "col_Vanta",
    "vip_Diamond",        # tier VIP
    "vip_Gold",
    "vip_Member",
    "vip_Platinum",
]

# on crée un DataFrame avec uniquement ces colonnes
X = clients[features].copy()

# on applique le StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# on remet en DataFrame pour garder les noms de colonnes
X_scaled = pd.DataFrame(X_scaled, columns=features)

print("Avant standardisation :")
print(X[["total_spent_log", "purchase_count", "age"]].describe().round(2))

print("\nAprès standardisation :")
print(X_scaled[["total_spent_log", "purchase_count", "age"]].describe().round(2))

# sauvegarde du dataset préprocessé
clients.to_csv(base + "clients_preprocessed.csv", index=False)
X_scaled.to_csv(base + "clients_ml_ready.csv", index=False)
print("\n✅ Fichiers sauvegardés : clients_preprocessed.csv et clients_ml_ready.csv")