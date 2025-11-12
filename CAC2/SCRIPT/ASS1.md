# COURS DE SCIENCE DES DONNÉES

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

## Détails techniques principaux :

Type : Multivarié, catégoriel

Nombre d'instances : 958 configurations possibles

Nombre de caractéristiques : 9 cases du plateau (top-left, top-middle, top-right, middle-left, middle-middle, middle-right, bottom-left, bottom-middle, bottom-right)

Valeurs des caractéristiques : chaque case peut contenir "x", "o" ou être vide ("b")

Variable cible : "class", indiquant victoire ("positive") ou non ("negative") pour "x"

Pas de valeurs manquantes dans les données

Algorithmes classiques (ID3, CN2, IB1, CITRE) donnent de bons résultats sur ce jeu de données
