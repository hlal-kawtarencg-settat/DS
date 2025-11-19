# Assignment MACHINE LEARNING

## HLAL KAWTAR
## N° 21008814


## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année



## Objectif et Fonctionnalité de ucimlrepo
L'objectif principal de la librairie ucimlrepo est de simplifier l'accès, le téléchargement et le chargement des jeux de données de l'UCI Machine Learning Repository directement dans un environnement Python, généralement sous forme de structures de données pandas DataFrame.

## Fonctionnalités Clés :
Accès Programmé aux Métadonnées : Elle permet de rechercher des jeux de données par ID ou nom, et de récupérer leurs métadonnées (description, nombre d'instances, nombre d'attributs, etc.) sans quitter votre code.

Téléchargement et Chargement Direct : La fonction principale vous permet de charger le jeu de données (les caractéristiques / features et la cible / target) directement sous forme de DataFrame, prêt pour l'analyse et la modélisation.

Structuration des Données : Elle s'occupe de la structure parfois hétérogène des jeux de données UCI, les présentant de manière uniforme.

## Exécution de la Commande
1. L'Outil : pip
pip est le gestionnaire de paquets (ou package installer) standard pour Python.

Son rôle est de télécharger des paquets (librairies) depuis le PyPI (Python Package Index), le répertoire officiel des logiciels Python, et de les installer dans votre environnement Python.

2. L'Action : install
L'argument install indique à pip de localiser et d'installer le paquet spécifié.

3. Le Paquet Cible : ucimlrepo
C'est le nom de la librairie à installer.

Processus d'Installation :
Recherche : pip contacte PyPI pour localiser le paquet ucimlrepo et ses métadonnées (dépendances, numéro de version).

Téléchargement : Il télécharge le paquet ucimlrepo ainsi que toutes les autres librairies dont ucimlrepo a besoin pour fonctionner (ses dépendances, comme potentiellement pandas, scikit-learn, etc.).

Installation : Les fichiers de la librairie et de ses dépendances sont copiés dans le dossier site-packages de l'environnement Python actif.

## Installer le package

```python
pip install ucimlrepo
```

## Importer les données

```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 
```
```python
import pandas as pd
import numpy as np

link = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

df = pd.read_csv(link, header="infer", delimiter=";")
print("\n========= Dataset summary ========= \n")
df.info()
print("\n========= A few first samples ========= \n")
print(df.head())
```
```python
X = df.drop("quality", axis=1) #we drop the column "quality"
Y = df["quality"]
print("\n========= Wine Qualities ========= \n")
print(Y.value_counts())
```
```python
# bad wine (y=0) : quality <= 5 and good quality (y= 1) otherwise
Y = [0 if val <=5 else 1 for val in Y]
```
```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
ax = plt.gca()
sns.boxplot(data=X,orient="v",palette="Set1",width=1.5, notch=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.figure()
corr = X.corr()
sns.heatmap(corr)
```
## classification des k


```python
from sklearn.model_selection import train_test_split
Xa, Xt, Ya, Yt = train_test_split(X, Y, shuffle=True, test_size=1/3,
stratify=Y)
Xa, Xv, Ya, Yv = train_test_split(Xa, Ya, shuffle=True, test_size=0.5,
stratify=Ya)
```
```python
from sklearn.neighbors import KNeighborsClassifier
# Fit the model on (Xa, Ya)
k = 3
clf = KNeighborsClassifier(n_neighbors = k)
clf.fit(Xa, Ya)
# Predict the labels of samples in Xv
Ypred_v = clf.predict(Xv)
# evaluate classification error rate
from sklearn.metrics import accuracy_score
error_v = 1-accuracy_score(Yv, Ypred_v)
```

```python
k_vector = np.arange(1, 37, 2) #define a vector of k=1, 3, 5, ...
error_train = np.empty(k_vector.shape)
error_val = np.empty(k_vector.shape)
for ind, k in enumerate(k_vector):
    #fit with k
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(Xa, Ya)
    # predict and evaluate on training and validation sets
    Ypred_train = clf.predict(Xa)
    error_train[ind] = 1 - accuracy_score(Ya, Ypred_train)
    Ypred_val = clf.predict(Xv)
    error_val[ind] = 1 - accuracy_score(Yv, Ypred_val)
```

```python
# some hints: get the min error and related k-value
err_min, ind_opt = error_val.min(), error_val.argmin()
k_star = k_vector[ind_opt]
```

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=True, with_std=True)
sc = sc.fit(Xa)
Xa_n = sc.transform(Xa)
Xv_n = sc.transform(Xv)
```

