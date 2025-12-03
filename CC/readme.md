# Breast Cancer

# A.LARHLIMI

## HLAL KAWTAR

<img src="image7.png" style="height:540px;margin-right:393px"/>

## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année

---

# PARTIE 1 : Présentation du Jeu de Données – Breast Cancer

## 1. Sélection du Dataset

Le dataset **Breast_Cancer.csv** contient des informations cliniques et démographiques concernant 4024 patientes atteintes d’un cancer du sein. Il regroupe un ensemble varié de variables telles que l’âge, les caractéristiques tumorales, les stades de la maladie, les récepteurs hormonaux, ainsi que des données de survie.

Ce dataset est typiquement utilisé dans des travaux de modélisation médicale, d’analyse de survie ou de prédiction d’évolution clinique. Sa richesse permet d’approfondir la compréhension des facteurs influençant la progression du cancer du sein.

## 2. Définition de la Problématique

La problématique principale abordée est la suivante :

**Quels facteurs cliniques et démographiques influencent la survie ou l’évolution du statut vital (Alive/Dead) chez des patientes atteintes d’un cancer du sein ?**

Cette problématique permet de poser les bases d’une analyse prédictive ou descriptive, visant à identifier les variables les plus déterminantes dans le pronostic des patientes.

## 3. Dictionnaire des Données (Metadata)

Le dataset comporte un ensemble de variables médicales structurées. Voici une description détaillée inspirée du modèle fourni.

### 3.1. Taille et Structure

* **Nombre d’observations :** 4024 entrées
* **Nombre de colonnes :** 16 variables
* **Format :** fichier CSV
* **Type d’étude :** données médicales réelles issues d’un registre ou d’une base clinique
* **Types de tâches possibles :** classification, analyse descriptive, prédiction de survie, clustering

### 3.2. Types de Variables

* **Variables catégorielles :** Race, Marital Status, T Stage, N Stage, 6th Stage, differentiate, Grade, A Stage, Estrogen Status, Progesterone Status, Status
* **Variables ordinales :** differentiate, Grade
* **Variables numériques :** Age, Tumor Size, Regional Node Examined, Reginol Node Positive, Survival Months
* **Cible (Target) potentielle :** Status (Alive / Dead) ou Survival Months

### 3.3. Description des principales colonnes

| Variable                   | Type                           | Signification                                     |
| -------------------------- | ------------------------------ | ------------------------------------------------- |
| **Age**                    | Numérique                      | Âge de la patiente au diagnostic.                 |
| **Race**                   | Catégorielle                   | Groupe racial déclaré.                            |
| **Marital Status**         | Catégorielle                   | Statut marital de la patiente.                    |
| **T Stage**                | Catégorielle                   | Taille / extension de la tumeur.                  |
| **N Stage**                | Catégorielle                   | Envahissement ganglionnaire.                      |
| **6th Stage**              | Catégorielle                   | Stade du cancer selon la 6e édition AJCC.         |
| **differentiate**          | Ordinale                       | Degré de différenciation tumorale.                |
| **Grade**                  | Ordinale                       | Grade histologique de la tumeur.                  |
| **A Stage**                | Catégorielle                   | Autre classification tumorale.                    |
| **Tumor Size**             | Numérique                      | Taille de la tumeur (mm).                         |
| **Estrogen Status**        | Catégorielle                   | Récepteurs d’œstrogènes : positifs / négatifs.    |
| **Progesterone Status**    | Catégorielle                   | Récepteurs de progestérone : positifs / négatifs. |
| **Regional Node Examined** | Numérique                      | Nombre de ganglions examinés.                     |
| **Reginol Node Positive**  | Numérique                      | Nombre de ganglions positifs.                     |
| **Survival Months**        | Numérique                      | Durée de survie en mois.                          |
| **Status**                 | Catégorielle (Target possible) | Statut vital final : Alive / Dead.                |

### 3.4. Identification de la Target

Les deux cibles les plus pertinentes selon l’objectif du projet sont :

 **Status** : pour prédire si une patiente est vivante ou décédée (classification)
   **Survival Months** : pour estimer la durée de survie (régression / analyse de survie)

La variable **Status** est la plus fréquemment utilisée dans les modèles prédictifs.

## Conclusion

Le dataset Breast Cancer est riche et varié, offrant de nombreuses possibilités analytiques allant de la classification à la modélisation de survie. La diversité des variables permet une compréhension approfondie des facteurs liés à la progression du cancer du sein, ouvrant la voie à des analyses robustes et pertinentes.


***

## 1. Pré-traitement (Preprocessing)

Le code dans le notebook a exécuté les étapes suivantes de pré-traitement :

### Nettoyage des données (Gestion des doublons, formatage)
* **Doublons** : Une vérification des lignes en double a été effectuée, et **aucun doublon** n'a été trouvé dans le jeu de données initial.
* **Formatage** : Des opérations de nettoyage ont été réalisées sur les valeurs de plusieurs colonnes catégorielles pour standardiser les entrées et gérer les erreurs de formatage (espaces inutiles), notamment dans :
    * `Grade` : Les valeurs ont été standardisées (par exemple, ' anaplastic; Grade IV' remplacé par '4') et le type de la colonne a été converti en numérique.
    * `Marital Status`, `Race`, `6th Stage`, `N Stage`, et `T Stage ` : Les espaces de début ou de fin ont été supprimés pour uniformiser les catégories.
    * Les valeurs non valides ('?') dans `Grade` et `differentiate` ont été remplacées par des valeurs manquantes (`np.nan`).

### Imputation des valeurs manquantes
* Les valeurs manquantes (`np.nan`) dans les colonnes `differentiate` et `Grade` ont été imputées en utilisant la **mode** (valeur la plus fréquente) de chaque colonne.
    * *Remarque* : L'imputation par la mode est une stratégie simple. Bien qu'elle soit efficace pour les données catégorielles ou discrètes, elle ne répond pas au critère d'une "stratégie avancée" (comme l'imputation par régression ou par K-Nearest Neighbors).

### Encodage des variables catégorielles
* Toutes les variables catégorielles (ex: `Race`, `Marital Status`, `T Stage`, `6th Stage`, `Estrogen Status`, `Status`) ont été transformées en valeurs numériques à l'aide du **Label Encoding** (`sklearn.preprocessing.LabelEncoder`).
    * *Note* : Le Label Encoding attribue un entier unique à chaque catégorie.

### Normalisation ou Standardisation des données numériques
* Les variables numériques (`Age`, `Grade`, `Tumor Size`, `Regional Node Examined`, `Reginol Node Positive`) ont été mises à l'échelle en utilisant la **Standardisation** (`sklearn.preprocessing.StandardScaler`).
    * La standardisation centre les données autour de zéro avec un écart type de un (méthode $Z-score$).
* **Gestion du déséquilibre de classe** : Après la séparation des variables indépendantes (`X`) et dépendante (`y` - 'Status'), un déséquilibre significatif a été identifié dans la variable cible. Le déséquilibre a été corrigé en utilisant la technique de suréchantillonnage **ADASYN (Adaptive Synthetic Sampling)** sur l'ensemble d'entraînement.

***

## 2. Analyse Exploratoire des Données (EDA)

### Visualisation des distributions (Histogrammes, Boxplots)
Plusieurs visualisations ont été générées pour explorer les distributions des variables clés:
* **Distribution d'Âge** (Histogramme) : La distribution d'âge est relativement uniforme, avec un pic autour de 50-60 ans.
* **Distribution de la Race** (Count Plot) : La catégorie 'White' est largement majoritaire dans le jeu de données.
* **Distribution du Grade** (Count Plot) : Le grade '2' est la catégorie la plus fréquente.
* **Distribution de la 6ème Étendue** (Count Plot) : 'IIA' est l'étendue la plus courante.
* **Taille de la Tumeur vs. Statut** (Boxplot) : Bien que les médianes soient similaires entre les patients 'Alive' et 'Dead', le groupe 'Dead' présente une plus grande dispersion des tailles de tumeur et plus d'aberrations avec des tumeurs très grandes.
* **Nœuds Régionaux Positifs vs. Statut** (Boxplot) : Le nombre médian de nœuds régionaux positifs est plus élevé pour le groupe 'Dead', ce qui suggère qu'un nombre plus élevé de nœuds positifs est un facteur de risque associé à un pronostic défavorable.

### Analyse des corrélations (Heatmap)
* Un **Heatmap** a été généré à partir de la matrice de corrélation pour visualiser les relations entre toutes les paires de variables.
* **Interprétation** : La variable cible `Status` (après encodage) montre une corrélation modérée positive avec `Reginol Node Positive` et `Tumor Size`, ainsi qu'une corrélation négative avec `Survival Months`. Cela confirme les observations des boxplots : plus le nombre de nœuds positifs et la taille de la tumeur sont élevés, plus le risque d'un statut "Dead" (1) est élevé, et moins les mois de survie sont nombreux.

### Feature Engineering
* **Statut** : La section EDA du notebook **ne contient pas d'étapes explicites de Feature Engineering** (création de nouvelles variables pertinentes à partir des caractéristiques existantes), se concentrant uniquement sur la visualisation et l'analyse de corrélation des caractéristiques d'origine (nettoyées et encodées).

***

## 3. Modélisation (Machine Learning)

### Stratégie de validation rigoureuse
* Le jeu de données a d'abord été divisé en ensembles d'entraînement et de test (`X_train`, `y_train`, `X_test`, `y_test`).
* La stratégie d'évaluation des modèles repose sur la **Cross-Validation à 10 plis (`cv=10`)** en utilisant `cross_val_score` pour chaque algorithme. Cette méthode assure une estimation robuste de la performance du modèle.

### Testez et comparez au moins trois algorithmes différents
Le code a testé et comparé **dix** algorithmes de classification différents:
1.  Logistic Regression (Régression Logistique)
2.  Decision Tree Classifier (Arbre de Décision)
3.  K-Nearest Neighbors (KNN)
4.  Gaussian Naive Bayes
5.  Multinomial Naive Bayes
6.  Support Vector Classifier (SVC)
7.  Random Forest Classifier
8.  XGBoost Classifier
9.  Multi-layer Perceptron (MLP)
10. Gradient Boosting Classifier

Les résultats de l'accuracy moyenne de la cross-validation sont les suivants:
| Algorithme | Accuracy Moyenne (Cross-Validation) |
| :--- | :--- |
| **Random Forest** | **0.876** |
| XGBoost | 0.863 |
| Decision Tree | 0.824 |
| K-Nearest Neighbors | 0.820 |
| Gradient Boosting | 0.775 |
| Multi-layer Perceptron | 0.751 |
| Logistic Regression | 0.722 |
| Support Vector Classifier | 0.711 |
| Gaussian Naive Bayes | 0.677 |
| Multinomial Naive Bayes | 0.620 |

**Conclusion de la comparaison** : L'algorithme **Random Forest** a obtenu la meilleure performance avec une *accuracy* moyenne de 87.6%.

### Optimisation des hyperparamètres
* **Statut** : Le notebook **ne contient aucune étape explicite d'optimisation des hyperparamètres** utilisant des méthodes comme `GridSearchCV` ou `RandomizedSearchCV`. Les modèles ont été entraînés soit avec les hyperparamètres par défaut, soit avec des valeurs choisies manuellement (par exemple, `n_neighbors=5` pour KNN).
