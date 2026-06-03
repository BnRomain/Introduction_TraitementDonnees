# Introduction au Traitement des Données — Projet Kaggle

**Cours :** Introduction to Data Processing — Lionel Fillatre, Polytech Nice Sophia (2025-2026)  
**Équipe :** Romain Ben, Loriziano Soave  
**Dates :** 1 au 5 juin 2026

---

## Question métier

> **Peut-on prédire si une expédition himalayenne va atteindre le sommet, à partir des caractéristiques connues avant le départ ?**

**Dataset :** [Himalayan Expeditions — Kaggle](https://www.kaggle.com/datasets/siddharth0935/himalayan-expeditions)  
11 000 expéditions, 65 colonnes. Variable cible : `success1` (booléen).  
**Type de modèle :** régression logistique (classification binaire : succès / échec).

---

## Calendrier

| Demi-journée | Date | Sujet |
|---|---|---|
| 1 | Lun. 1er juin matin | Choix du dataset + définition du problème métier |
| 2 | Lun. 1er juin après-midi | Prise en main des données (wrangling) |
| 3 | Mar. 2 juin matin | Visualisation des données + data storytelling |
| 4 | Mar. 2 juin après-midi | Pipeline de données, feature engineering, intro régression |
| 5 | Mer. 3 juin matin | Feature engineering + visualisation |
| 6 | Mer. 3 juin après-midi | Régression logistique + storytelling |
| 7 | Jeu. 4 juin matin | Pipeline final + préparation soutenance |
| 8 | Ven. 5 juin matin | Soutenance orale + rapport |

**Slides à déposer sur Moodle** : jeudi soir (4 juin)

---

## Livrables

### 1. Soutenance orale (5 min + 5 min Q&A)
Exactement **5 slides**, format **data storytelling** :

1. **Business goal** — présenter le dataset et la question
2. **Data description** — les features retenues, leur type, les valeurs manquantes
3. **Handcrafted features** — les features créées et leur justification
4. **Régression logistique** — résultats, métriques, interprétation
5. **Conclusion** — ce que l'analyse révèle, pistes d'amélioration

### 2. Rapport écrit (10 pages max)

| Section | Pages max |
|---|---|
| 1. Business goal | 1 |
| 2. Team management | 1 |
| 3. Data visualisation | 2 |
| 4. Handcrafted features | 2 |
| 5. Régression logistique | 2 |
| 6. Conclusion | 1 |
| 7. Références | 1 |

---

## Étapes techniques

### Étape 1 — Sélection des variables (fait)

Sur les 65 colonnes, on ne garde que des **variables connues avant le départ** (pas de data leakage).

**Variables exclues car post-expédition (data leakage) :**
- `smtmembers`, `smthired`, `smtdate`, `smttime`, `smtdays` — infos post-sommet
- `mdeaths`, `hdeaths` — décès survenus pendant
- `termreason`, `termnote`, `termdate` — raison de fin d'expédition
- `highpoint` — altitude max atteinte (= sommet si succès)
- `success2/3/4`, `ascent1/2/3/4` — variantes du résultat

**Variables candidates à retenir (max 10) :**

| Variable | Type | Description |
|---|---|---|
| `year` | numérique | Année de l'expédition |
| `season` | catégoriel | Saison (1=hiver, 2=printemps, 3=été, 4=automne) |
| `peakid` | catégoriel | Identifiant du sommet visé |
| `totmembers` | numérique | Nombre total de membres |
| `tothired` | numérique | Nombre de Sherpas/porteurs engagés |
| `o2used` | booléen | Utilisation d'oxygène prévu |
| `camps` | numérique | Nombre de camps installés |
| `comrte` | booléen | Route commerciale (oui/non) |
| `nohired` | booléen | Expédition sans personnel engagé |
| `nation` | catégoriel | Nationalité de l'équipe |

La sélection finale se fait après l'exploration (étape 2).

---

### Étape 2 — Data Wrangling

C'est la partie la plus longue. Dans l'ordre :

**Sanity checks :**
- Charger le CSV et afficher les premières lignes (`df.head()`)
- Vérifier les types (`df.dtypes`)
- Compter les valeurs manquantes (`df.isnull().sum()`)
- Chercher les doublons (`df.duplicated().sum()`)
- Vérifier que `success1` est bien un booléen et regarder l'équilibre des classes

**Nettoyage :**
- Garder uniquement les colonnes retenues + la cible `success1`
- Gérer les valeurs manquantes (suppression ou imputation selon la colonne)
- Encoder les variables catégorielles (`season`, `peakid`, `nation`)

**Transformation :**
- Standardiser les variables numériques (`totmembers`, `tothired`, `camps`, `year`)

**Bibliothèques :** `pandas`, `numpy`

---

### Étape 3 — Visualisation

Produire des figures qui **racontent quelque chose** sur la difficulté d'atteindre un sommet.

**Idées de figures pertinentes :**
- Taux de succès par saison (bar chart)
- Taux de succès par sommet — les 10 plus tentés (bar chart horizontal)
- Évolution du taux de succès par année (line chart)
- Distribution de la taille des équipes (histogramme, succès vs échec)
- Impact de l'utilisation d'oxygène sur le succès (bar chart)

**Règles :** axe Y à zéro, titre + légendes toujours présents, une figure = un message.

**Bibliothèques :** `matplotlib`, `seaborn`

---

### Étape 4 — Feature Engineering

Créer de nouvelles features à partir des variables brutes.

**Idées à explorer :**
- `ratio_hired` = `tothired / totmembers` (proportion de l'équipe qui est du personnel engagé)
- `era` = période historique par binning sur `year` (avant 1970 / 1970-2000 / après 2000)
- `big_team` = booléen si `totmembers` > seuil (à déterminer visuellement)
- Encodage ordinal de `season` selon le taux de succès historique par saison

Pour chaque feature créée : visualiser sa relation avec `success1` avant de la garder.

---

### Étape 5 — Régression logistique

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
```

**Pipeline :**
1. Séparer X (features) et y (`success1`)
2. Split train/test (80/20)
3. Standardiser les features numériques (fit sur train uniquement)
4. Entraîner `LogisticRegression()`
5. Évaluer sur le test set : accuracy, precision, recall, matrice de confusion

**Questions à se poser sur les résultats :**
- Le modèle est-il meilleur qu'un classifieur naïf (toujours prédire la classe majoritaire) ?
- Quels coefficients ont le plus de poids ? Est-ce cohérent intuitivement ?
- Y a-t-il du sur-apprentissage (performance train >> test) ?

---

## Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── DataBase/
│   └── exped.csv
├── Cours/
│   ├── Cours1.pdf
│   ├── Cours2.pdf
│   ├── Cours3.pdf
│   └── Cours4.pdf
├── 01_exploration.py
├── 02_wrangling.py
├── 03_visualisation.py
├── 04_feature_engineering.py
├── 05_regression.py
├── figures/
│   └── (figures exportées via plt.savefig pour le rapport)
└── report/
    └── (rapport PDF + slides)
```

**Installer l'environnement :** `py -m pip install -r requirements.txt`  
**Lancer un script :** `py 01_exploration.py` (ou le bouton ▶ dans VS Code)

---

## Ressources utiles

- [Dataset Kaggle — Himalayan Expeditions](https://www.kaggle.com/datasets/siddharth0935/himalayan-expeditions)
- [Notebooks Kaggle sur ce dataset](https://www.kaggle.com/datasets/siddharth0935/himalayan-expeditions/code) (s'en inspirer, pas copier)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn — Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Scikit-learn — Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Seaborn gallery](https://seaborn.pydata.org/examples/index.html)

---

## Notes

- S'inspirer des notebooks Kaggle existants est encouragé, mais comprendre et adapter le code est obligatoire
- Format "data storytelling" exigé : chaque figure doit raconter quelque chose, pas juste exister
- L'URL de ce dépôt Git doit figurer dans le rapport
