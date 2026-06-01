# =============================================================
# 01 - Exploration des donnees
#
# Question metier : peut-on predire si une expedition himalayenne
# atteint le sommet (success1) a partir d'infos connues AVANT le depart ?
#
# But de ce script : REGARDER les donnees brutes avant de toucher a quoi
# que ce soit. On ne nettoie pas encore, on ne modelise pas.
#
# Inspiration : notebook de Muhammed Ali Yilmaz, "Himalayan Climb Prediction
# with ML/DL" (https://www.kaggle.com/code/muhammedaliyilmazz/himalayan-climb-prediction-with-ml-dl).
# On s'en inspire pour le nettoyage et les idees de features, mais on
# construit notre propre regression logistique.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement (fourni). Le chemin part de la racine du projet.
df = pd.read_csv("DataBase/exped.csv", low_memory=False)
print("Lignes:", df.shape[0], "| Colonnes:", df.shape[1])


# -------------------------------------------------------------
# 1. PREMIER REGARD
# A vous. Affichez les premieres lignes pour voir a quoi ressemblent
# les donnees. Utilisez print(df.head()).
# Question : qu'est-ce qui est lisible (year, season...) et qu'est-ce
# qui ressemble a un code interne (expid, chksum...) ?
# -------------------------------------------------------------



# -------------------------------------------------------------
# 2. TYPES ET VALEURS MANQUANTES
# A vous. Pour chaque colonne : son type et combien de NaN.
# Pistes : df.dtypes, df.info(), df.isnull().sum()
# Pour trier les colonnes les plus vides en premier :
#   print(df.isnull().sum().sort_values(ascending=False))
# Question : quelles colonnes sont quasi vides et donc inutilisables ?
# -------------------------------------------------------------



# -------------------------------------------------------------
# 3. LA CIBLE : success1
# A vous. C'est ce qu'on veut predire. Verifiez :
#   - son type        : df["success1"].dtype
#   - ses valeurs     : df["success1"].unique()
#   - sa repartition  : df["success1"].value_counts()
#                       df["success1"].value_counts(normalize=True)  -> en %
# Question cle : quelle proportion d'expeditions reussissent ?
# (si tres desequilibre, ex 90/10, on en tiendra compte a la regression)
# -------------------------------------------------------------



# -------------------------------------------------------------
# 4. SELECTION DES COLONNES (max 10)
# Etape qui demande le plus de reflexion. Deux regles :
#
# Regle 1 - PAS DE DATA LEAKAGE : on ne garde QUE des infos connues
# AVANT le depart. Interdit (post-expedition) :
#   smtmembers, smthired, smtdate, smttime, smtdays, mdeaths, hdeaths,
#   termreason, termnote, termdate, highpoint, success2/3/4, ascent1/2/3/4
#
# Regle 2 - PERTINENCE : pour chaque colonne, demandez-vous si un
# alpiniste experimente penserait qu'elle influence les chances de reussite.
#
# A vous. Definissez la liste des ~10 colonnes, puis le sous-tableau :
#   colonnes = ["year", "season", "peakid", "totmembers", "tothired", "o2used", "camps"]  # a completer
#   df_select = df[colonnes + ["success1"]]
#   print(df_select.head())
# Justifiez chaque choix en commentaire (reutilisable dans le rapport).
# -------------------------------------------------------------


