# Nous travaillons sur des expéditions dans l'himalaya. Le dataset que nous utlisons repertorie toutes les expéditions dans l'himalaya qui ont eu lieu depuis 1900.
# Notre bute est de repondre à la question "Peut-on prédire si une expédition va atteindre le sommet ?".

# Ce code est inspiré du travail de Muhammed Ali YILMAZ sur son notebook "Himalayan Climb Prediction with ML/DL".
# Lien : "https://www.kaggle.com/code/muhammedaliyilmazz/himalayan-climb-prediction-with-ml-dl/notebook"

# Import des librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


