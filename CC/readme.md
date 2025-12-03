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
