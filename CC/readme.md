# La Mortalité liée à la Covid-19 en France

# A.LARHLIMI

## HLAL KAWTAR

<img src="image7.png" style="height:540px;margin-right:393px"/>

## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année

---

# PARTIE 1 : Présentation des Données du Jeu de Données

## 1. Sélection du Dataset

Le jeu de données utilisé dans ce projet correspond à l'évolution quotidienne du nombre de décès liés au COVID-19 en France. Il est extrait d'une source statistique publique et se présente sous la forme d'un fichier CSV nommé **FR_covid_deaths.csv**. Celui-ci regroupe des observations temporelles couvrant une période de 24 jours consécutifs.

L'objectif principal de ce dataset est d'analyser la dynamique de mortalité durant la période étudiée, en vue d'une modélisation ou d'une compréhension statistique de l'évolution du phénomène.

## 2. Définition de la Problématique

La problématique associée à ce jeu de données est la suivante :

**Comment analyser et modéliser l'évolution quotidienne du nombre de décès liés au COVID-19 en France, afin d'identifier les tendances observables et de permettre une éventuelle prédiction ?**

Cette problématique s'inscrit dans un contexte d'analyse de séries temporelles, où l'objectif est d'explorer les patterns, les variations et les comportements du phénomène observé.

## 3. Dictionnaire des Données (Metadata)

Le dataset contient deux variables principales organisées sous forme de série temporelle. Voici une description structurée calquée sur le modèle demandé.

### 3.1. Taille et Structure

* **Nombre d’observations :** 24 entrées
* **Format :** fichier CSV
* **Type d’étude :** données factuelles (statistiques officielles)
* **Types de tâches possibles :** analyse descriptive, modélisation de séries temporelles, prévision

### 3.2. Types de Variables

* **Variables temporelles :** dates (format texte)
* **Variables numériques :** nombre quotidien de décès
* **Cible (Target) :** variable numérique (nombre de décès)

### 3.3. Description des principales colonnes

| Variable | Type               | Signification                                                |
| -------- | ------------------ | ------------------------------------------------------------ |
| **ds**   | Temporelle / Texte | Correspond à la date d’observation (format jour/mois/année). |
| **y**    | Numérique          | Nombre de décès enregistrés à la date correspondante.        |

### 3.4. Identification de la Target

La variable cible retenue pour l’analyse et la modélisation est :  **"y" — le nombre quotidien de décès.**

Cette variable représente directement l’évolution du phénomène étudié et constitue l’élément central de toute analyse prédictive.

## Conclusion

Le jeu de données étudié se présente sous la forme d'une série temporelle simple, composée de deux variables essentielles : la date et le nombre de décès. Sa structure réduite et claire permet une première exploration efficace et ouvre la voie à une modélisation statistique ou machine learning. Cette analyse constitue une première étape essentielle avant l'application de techniques de visualisation ou de prédiction.
