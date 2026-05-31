# Introduction au Traitement des Données — Projet Kaggle

**Cours :** Introduction to Data Processing — Lionel Fillatre, Polytech Nice Sophia (2025-2026)  
**Équipe :** Romain Ben, Loriziano Soave  
**Dates :** 1 au 5 juin 2026

---

## Objectif du projet

Choisir un dataset Kaggle, définir une **question métier** claire, puis mener une analyse complète de la donnée pour y répondre. Le but n'est pas de répondre parfaitement à la question, mais de montrer que l'analyse des données éclaire la réponse.

Contraintes sur le dataset :
- Chaque équipe a son propre dataset (pas de doublon entre équipes)
- Le dataset Titanic est **interdit**
- Le dataset doit être pertinent pour une régression (linéaire, logistique ou softmax)

---

## Calendrier

| Demi-journée | Date | Sujet |
|---|---|---|
| 1 | Lun. 1er juin matin | Choix du dataset + définition du problème métier |
| 2 | Lun. 1er juin après-midi | Prise en main des données (wrangling) |
| 3 | Mar. 2 juin matin | Visualisation des données + data storytelling |
| 4 | Mar. 2 juin après-midi | Pipeline de données, feature engineering, intro régression |
| 5 | Mer. 3 juin matin | Feature engineering + visualisation |
| 6 | Mer. 3 juin après-midi | Régression linéaire + storytelling |
| 7 | Jeu. 4 juin matin | Pipeline final + préparation soutenance |
| 8 | Ven. 5 juin matin | Soutenance orale + rapport |

**Slides à déposer sur Moodle** : jeudi soir (4 juin)

---

## Livrables

### 1. Soutenance orale (5 min + 5 min Q&A)
Exactement **5 slides**, une par thème, format **data storytelling** (narrative claire et engageante) :

1. **Business goal** — quel est le dataset ? pourquoi est-il utile ?
2. **Data description** — combien de features ? quels types ?
3. **Handcrafted features** — quelles features avez-vous créées et pourquoi ?
4. **Régression** — quelle régression ? pourquoi ? résultats ?
5. **Conclusion** — bénéfices de l'analyse, pistes d'amélioration

### 2. Rapport écrit (10 pages max)

| Section | Pages max |
|---|---|
| 1. Business goal | 1 |
| 2. Team management (organisation, planning) | 1 |
| 3. Data visualisation (figures commentées) | 2 |
| 4. Handcrafted features (description + justification) | 2 |
| 5. Régression (choix + résultats + fiabilité) | 2 |
| 6. Conclusion | 1 |
| 7. Références | 1 |

---

## Étapes techniques

### Étape 1 — Choix du dataset et du problème

- Parcourir [kaggle.com/datasets](https://www.kaggle.com/datasets) pour trouver un dataset intéressant
- Définir une question métier : *"Peut-on prédire X à partir de Y ?"*
- Vérifier que le dataset est adapté à la régression (variable cible continue, binaire ou multiclasse)
- S'assurer qu'aucune autre équipe n'a pris ce dataset

**Questions à se poser :**
- Combien de lignes et de colonnes ?
- Y a-t-il des valeurs manquantes ?
- Quelle est la variable cible (ce qu'on veut prédire) ?

---

### Étape 2 — Data Wrangling (Cours 2)

C'est souvent la partie la plus longue du travail d'un data scientist. Elle inclut :

**Sanity checks initiaux :**
- Ouvrir le fichier et regarder son contenu brut
- Vérifier les types de colonnes (numérique, texte, date, catégoriel)
- Repérer les valeurs nulles / manquantes
- Identifier les doublons
- Vérifier la cohérence des valeurs (valeurs aberrantes, erreurs de saisie)

**Nettoyage :**
- Supprimer ou imputer les valeurs manquantes
- Renommer les colonnes si nécessaire
- Supprimer les colonnes inutiles

**Transformation des features :**
- Convertir les unités si besoin
- Normalisation (imposer un min/max) ou standardisation (moyenne 0, écart-type 1)
- Encodage des variables catégorielles

**Bibliothèques principales :** `pandas`, `numpy`

---

### Étape 3 — Visualisation des données (Cours 3)

L'objectif est de produire des **figures claires et honnêtes** qui racontent quelque chose sur vos données.

**Règles de base à respecter :**
- L'axe Y commence toujours à zéro (sauf raison explicite)
- Titre, axes et légendes toujours présents
- Ne pas sélectionner seulement les données qui "arrangent" la narrative
- Une figure = un message clair

**Types de visualisations à connaître :**
- Distribution d'une variable : histogramme, boxplot, violin plot
- Relation entre deux variables continues : scatter plot
- Variable catégorielle vs continue : bar chart, boxplot groupé
- Corrélations : heatmap de corrélation

**Bibliothèques principales :** `matplotlib`, `seaborn`

---

### Étape 4 — Feature Engineering (Cours 2 + 4)

Créer de nouvelles features à partir des features existantes, grâce à la connaissance du domaine.

**Techniques vues en cours :**
- **Binning** : transformer une variable continue en catégories discrètes (ex : âge -> tranche d'âge)
- **Agrégation** : regrouper des catégories rares sous un label commun (ex : "Rare")
- **Interaction** : créer une feature produit de deux features (ex : `Pclass * Age`)
- **Encodage ordinal non-linéaire** : assigner des valeurs non uniformes à des catégories ordonnées

**À chaque feature créée, se demander :**
- Pourquoi cette feature pourrait aider le modèle ?
- Est-ce qu'elle crée une dépendance avec la variable cible ?
- Est-ce qu'on peut la visualiser pour vérifier sa pertinence ?

---

### Étape 5 — Modélisation (Cours 4)

**Quel type de régression choisir ?**

| Type | Variable cible | Exemple |
|---|---|---|
| Régression linéaire | Continue (ex : prix, température) | Prédire le prix d'une maison |
| Régression logistique | Binaire (0 ou 1) | Prédire si un client va partir |
| Régression softmax | Multiclasse (3+ catégories) | Prédire la classe d'une fleur |

**Pipeline recommandé :**
1. Séparer les données en train / test
2. Standardiser les features numériques (fit sur train, transform sur train + test)
3. Entraîner le modèle sur le train set
4. Évaluer sur le test set

**Bibliothèque principale :** `scikit-learn`
- Régression linéaire : `sklearn.linear_model.LinearRegression`
- Régression logistique : `sklearn.linear_model.LogisticRegression`
- Régression softmax : `LogisticRegression(multi_class='multinomial')`
- Validation : `train_test_split`, métriques selon le type (MSE, accuracy, etc.)

**Validation du modèle :**
- Les résultats sont-ils cohérents ? (pas de 100% accuracy sur un vrai dataset)
- Le modèle généralise-t-il ? (comparer performance train vs test)
- Les features les plus importantes font-elles sens intuitivement ?

---

## Structure du dépôt

```
.
├── README.md
├── data/
│   └── (dataset brut téléchargé depuis Kaggle)
├── notebooks/
│   ├── 01_exploration.ipynb       # Premiers regards sur les données
│   ├── 02_wrangling.ipynb         # Nettoyage et transformation
│   ├── 03_visualisation.ipynb     # Figures et data storytelling
│   ├── 04_feature_engineering.ipynb
│   └── 05_regression.ipynb
└── report/
    └── (rapport PDF + slides)
```

---

## Ressources utiles

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Seaborn gallery](https://seaborn.pydata.org/examples/index.html)

---

## Notes

- Réutiliser des notebooks Kaggle existants est encouragé, mais les comprendre et les adapter est obligatoire
- Le format "data storytelling" est exigé : chaque figure doit raconter quelque chose, pas juste exister
- L'URL de ce dépôt Git doit figurer dans le rapport
