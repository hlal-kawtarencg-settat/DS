# Assignment DE SCIENCE DES DONNÉES

## HLAL KAWTAR
## N° 21008814

<img src="WhatsApp Image 2025-10-29 at 11.37.52.png" style="height:1092px;margin-right:1062px"/>

## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année





L’objectif principal de l’étude basée sur le jeu de données "Tic-Tac-Toe Endgame" est d’explorer la possibilité de prédire automatiquement, à partir de la configuration finale d’un plateau de morpion, si le joueur "x" a gagné la partie ou non.

## Objectifs spécifiques

- Développer et tester des algorithmes de classification supervisée capables d’apprendre à détecter une victoire de "x" en se basant sur les positions du plateau à la fin du jeu.[1]
- Comparer l’efficacité de différents modèles d’apprentissage automatique (arbres de décision, instance-based learning, algorithmes à règles) sur une tâche symbolique, simple et bien définie.
- Fournir un exemple pédagogique illustrant la prise de décision dans des systèmes basés sur des données catégorielles, avec la possibilité de valider les modèles grâce à un ensemble de données exhaustif et sans valeurs manquantes.
- Étudier la généralisabilité et la robustesse de méthodes classiques sur un problème où toutes les configurations possibles sont représentées.

Cette approche permet de mieux comprendre le comportement des algorithmes sur des tâches simples et de servir de cas d’étude introductif à la classification pour l’apprentissage automatique.


## Étapes principales du code

- **Installation du package**  
La ligne `pip install ucimlrepo` installe la bibliothèque qui facilite l’accès direct aux jeux de données de la UCI via Python.

- **Importation du module**  
`from ucimlrepo import fetch_ucirepo` importe la fonction nécessaire pour récupérer un dataset UCI selon son identifiant.

- **Téléchargement et chargement du dataset**  
`tic_tac_toe_endgame = fetch_ucirepo(id=101)` récupère le dataset Tic-Tac-Toe Endgame (identifiant 101) et le stocke dans une variable Python.  
Les données sont structurées dans un objet comportant plusieurs attributs utiles.

- **Extraction des données sous forme de DataFrames**  
`X = tic_tac_toe_endgame.data.features` extrait les caractéristiques (les cases du plateau) sous forme de DataFrame pandas.  
`y = tic_tac_toe_endgame.data.targets` extrait la variable cible (victoire ou non de "x") sous forme de DataFrame.

- **Affichage des métadonnées du dataset**  
`print(tic_tac_toe_endgame.metadata)` affiche les informations générales du dataset : thème, nombre d'instances, nombre de variables, liens vers la source, licence, etc.

- **Affichage de la description des variables**  
`print(tic_tac_toe_endgame.variables)` affiche les détails sur chaque colonne du dataset : nom, type, rôle (feature ou target), valeurs possibles et description.

## Utilité et lecture des sorties

- Ce code permet de charger immédiatement les données pour les analyser, modéliser ou visualiser dans un environnement Python.
- Les sorties des commandes `print(...)` sont essentielles pour comprendre la structure du jeu de données et préparer une analyse ou modélisation machine learning adaptée.


```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
tic_tac_toe_endgame = fetch_ucirepo(id=101) 
  
# data (as pandas dataframes) 
X = tic_tac_toe_endgame.data.features 
y = tic_tac_toe_endgame.data.targets 
  
# metadata 
print(tic_tac_toe_endgame.metadata) 
  
# variable information 
print(tic_tac_toe_endgame.variables) 
```

## Détails techniques principaux :

Type : Multivarié, catégoriel

Nombre d'instances : 958 configurations possibles

Nombre de caractéristiques : 9 cases du plateau (top-left, top-middle, top-right, middle-left, middle-middle, middle-right, bottom-left, bottom-middle, bottom-right)

Valeurs des caractéristiques : chaque case peut contenir "x", "o" ou être vide ("b")

Variable cible : "class", indiquant victoire ("positive") ou non ("negative") pour "x"

Pas de valeurs manquantes dans les données

Algorithmes classiques (ID3, CN2, IB1, CITRE) donnent de bons résultats sur ce jeu de données

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for the plots
sns.set_style('whitegrid')

# 1. Distribution of the target variable (win/loss for 'x')
plt.figure(figsize=(6, 4))
sns.countplot(data=y, x='class', palette='viridis')
plt.title('Distribution of Game Outcomes (Win for X)')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()
```
<img src="téléchargement.png" style="height:540px;margin-right:393px"/>

## Distribution of Each Board Position
Let's visualize the distribution of 'x', 'o', or 'b' (blank) for each square on the Tic-Tac-Toe board. This helps us understand common patterns in the endgame configurations.


```python
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharey=True)
axes = axes.flatten()

for i, column in enumerate(X.columns):
    sns.countplot(data=X, x=column, ax=axes[i], palette='magma', order=['x', 'o', 'b'])
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()
```
<img src="téléchargement (1).png" style="height:540px;margin-right:393px"/>

## Relationship between a Feature and the Target

Let's look at how one specific square's state relates to the final game outcome. We'll examine the 'middle-middle-square' as an example.

```python
# Combine X and y for easier plotting of relationships
df_combined = X.copy()
df_combined['class'] = y['class']

plt.figure(figsize=(8, 6))
sns.countplot(data=df_combined, x='middle-middle-square', hue='class', palette='viridis')
plt.title('Game Outcome by Middle-Middle Square State')
plt.xlabel('Middle-Middle Square State')
plt.ylabel('Count')
plt.legend(title='Outcome')
plt.show()
```
<img src="téléchargement (2).png" style="height:540px;margin-right:393px"/>
