# =============================================================
# 01 - Exploration des donnees
#
# Question metier : peut-on predire si une expedition himalayenne
# atteint le sommet (success) a partir d'infos connues AVANT le depart ?
#
# POINT DE DEPART : on part directement des ~20 features principales
# retenues par Muhammed Ali Yilmaz dans son notebook Kaggle (autorise :
# on a le droit d'utiliser la contribution des autres). On documente et
# on nettoie ces donnees deja reduites, plutot que les 65 colonnes brutes.
#
# Inspiration : "Himalayan Climb Prediction with ML/DL"
# https://www.kaggle.com/code/muhammedaliyilmazz/himalayan-climb-prediction-with-ml-dl
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Chargement du dataset brut (65 colonnes) -----
df_brut = pd.read_csv("DataBase/exped.csv", low_memory=False)
print("Dataset brut :", df_brut.shape[0], "lignes,", df_brut.shape[1], "colonnes")

# ----- Reduction aux 20 features de Muhammed + la cible -----
# Cle = nom reel dans le CSV, valeur = nom lisible (dictionnaire gemini-code.md).
# Le renommage est une etape de wrangling (Cours 2 : "Relabelling Columns").
correspondance = {
    "year": "year",
    "season": "season",
    "peakid": "peakid",
    "totmembers": "totmembers",
    "smtmembers": "smtmembers",
    "mdeaths": "mdeaths",
    "tothired": "tothired",
    "smthired": "smthired",
    "hdeaths": "hdeaths",
    "comrte": "comrte",
    "o2used": "o2used",
    "bcdate": "bcdate",
    "smtdate": "smtdate",
    "termdate": "termdate",
    "totdays": "totdays",
    "camps": "camps",
    "rope": "rope",
    "host": "host_country",
    "nation": "expedition_nation",
    "route1": "primary_route",
    "success1": "success",
}

df = df_brut[list(correspondance.keys())].rename(columns=correspondance)
print("Dataset reduit :", df.shape[0], "lignes,", df.shape[1], "colonnes")

# -------------------------------------------------------------
# ATTENTION - DATA LEAKAGE
# Parmi ces 20 features, certaines decrivent ce qui s'est passe PENDANT
# ou APRES l'expedition. Elles ne pourront PAS servir de variables
# predictives dans la regression (on predirait le resultat avec le
# resultat). On les garde pour la description / le storytelling, mais
# elles seront exclues du modele :
#   smtmembers, mdeaths, smthired, hdeaths, smtdate, termdate, totdays
#
# Features utilisables comme predicteurs (connues avant le depart) :
#   year, season, peakid, totmembers, tothired, comrte, o2used,
#   bcdate, camps, rope, host_country, expedition_nation, primary_route
# (Le modele final devra retenir AU MAXIMUM 10 de ces features.)
# -------------------------------------------------------------


# -------------------------------------------------------------
# 1. PREMIER REGARD
# A vous. print(df.head()) pour voir a quoi ressemblent les donnees reduites.
# -------------------------------------------------------------



# -------------------------------------------------------------
# 2. TYPES ET VALEURS MANQUANTES
# A vous. Sur le df REDUIT :
#   print(df.dtypes)
#   print(df.isnull().sum().sort_values(ascending=False))
# Question : quelles colonnes ont beaucoup de NaN ? Comment les traiter
# au nettoyage (suppression de lignes ? imputation ? suppression colonne) ?
# -------------------------------------------------------------



# -------------------------------------------------------------
# 3. LA CIBLE : success
# A vous. Verifiez le type, les valeurs, et surtout la repartition :
#   print(df["success"].value_counts(normalize=True))
# Question cle : quelle proportion d'expeditions reussissent ?
# -------------------------------------------------------------



# -------------------------------------------------------------
# 4. CHOIX DES FEATURES DU MODELE (max 10)
# A vous. Parmi les features utilisables (voir bloc DATA LEAKAGE ci-dessus),
# choisissez-en au maximum 10 pour le futur modele. Justifiez chaque choix
# en commentaire (reutilisable dans le rapport, section "Data description").
# Exemple :
#   features_modele = ["year", "season", "peakid", "totmembers", "tothired",
#                      "o2used", "camps", "host_country"]  # a discuter a deux
# -------------------------------------------------------------


