# STUDENT STRESS ANALYSIS

# A.LARHLIMI

## HLAL KAWTAR

<img src="image7.png" style="height:540px;margin-right:393px"/>

## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année


## Description
Ce jeu de données a été créé pour comprendre le niveau de stress des étudiants. Il contient les réponses de 53 étudiants qui ont évalué leur qualité de sommeil, leurs maux de tête, leurs performances académiques, leur charge d'études ou leurs activités extrascolaires. L'objectif est de découvrir comment ces facteurs affectent le niveau de stress des étudiants et d'identifier ceux qui contribuent le plus au stress.

## Contenu
Chacune des sept colonnes de ce jeu de données, qui comprend les réponses de 53 étudiants, représente une évaluation ou une fréquence concernant les maux de tête, les performances académiques, la charge d'études, la qualité du sommeil, les activités extrascolaires et le niveau de stress général. Toutes les données sont collectées à l'aide d'un formulaire d'auto-évaluation simple.

## Contexte
Le jeu de données se concentre sur la santé mentale et le bien-être des étudiants. Il facilite l'analyse de la manière dont les niveaux de stress des étudiants sont influencés par les choix de mode de vie (tels que les loisirs et les habitudes de sommeil) et la pression académique. Il peut être utilisé par des chercheurs ou des analystes de données pour étudier les corrélations et les schémas de stress, ainsi que pour faire des recommandations sur les moyens d'améliorer la santé mentale des étudiants.

## Phase de Préparation et Configuration
## Importation des bibliothèques

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

**pandas (pd)** : Bibliothèque pour la manipulation de données structurées (tableaux, DataFrames)

**numpy (np)** : Bibliothèque pour les calculs numériques et les opérations sur les matrices

**matplotlib.pyplot (plt)** : Outil de visualisation pour créer des graphiques et des diagrammes

**seaborn (sns)** : Bibliothèque de visualisation statistique construite sur matplotlib, offrant des graphiques plus sophistiqués

**warnings.filterwarnings("ignore")** : Supprime les avertissements pour une sortie plus propre
