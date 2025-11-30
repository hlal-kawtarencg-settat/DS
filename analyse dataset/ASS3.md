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


## Le Phénomène du Stress Étudiant : Un Impératif de Santé Publique
Le stress chez les étudiants universitaires représente un enjeu sociétal et de santé publique croissant à l'échelle mondiale. Ce phénomène ne se limite pas à une difficulté académique, mais constitue un facteur de risque majeur affectant le bien-être général, le comportement et la réussite potentielle personnelle et professionnelle. Le contexte académique, marqué par le passage du lycée à l'université, l'augmentation de la charge de travail et l'adaptation à un nouvel environnement, expose particulièrement cette population à des pressions psychosociales élevées. 

La gravité de la situation est illustrée par le fait que le suicide est la deuxième cause de mortalité chez les jeunes de 15 à 24 ans. Face à cette crise, la détection précoce des comportements et des pensées suicidaires, ainsi que l'identification des étudiants en détresse psychologique, sont primordiales pour fournir une prise en charge adéquate. L'Intelligence Artificielle (IA), notamment via l'apprentissage automatique (Machine Learning, ML), est de plus en plus utilisée pour modéliser et prédire ces niveaux de stress , offrant un potentiel de révolutionner la psychiatrie en permettant des diagnostics plus précis et des évaluations pronostiques plus fiables. Toutefois, cette application nécessite une méthodologie rigoureuse et une évaluation critique des modèles. 

## Cadre Technique de la Tâche : Classification Supervisée
La prédiction du niveau de stress à partir de données d'enquête ou de mesures physiologiques s'inscrit typiquement dans le cadre d'une tâche de classification supervisée. L'objectif est de catégoriser les étudiants soit de manière binaire (stress/non-stress), soit en classes multiples (faible, modéré, élevé).   

La pipeline d'apprentissage automatique employée pour cette tâche implique plusieurs étapes structurées. Elle commence par la collecte de données, souvent multimodales, intégrant des indicateurs physiologiques (comme la fréquence cardiaque ou la tension artérielle ) et des données d'enquête psychosociales. Suit le prétraitement des données, incluant notamment la gestion des variables catégorielles. Ensuite, la phase de modélisation utilise des algorithmes sophistiqués, le Random Forest étant un choix fréquent et performant dans ce domaine. Enfin, l'évaluation et l'interprétation du modèle sont réalisées grâce à des métriques précises et des outils d'Explicabilité (XAI), tels que SHAP. Pour garantir la fiabilité de l'application de l'IA en santé mentale (SMH), une démarche éthique et méthodologique est indispensable dès la conception de la recherche.


## Jeu de données utilisé
Le jeu de données étudié comporte 520 observations et 6 variables (5 caractéristiques explicatives et 1 variable cible). Chaque observation correspond à un étudiant ayant répondu à un questionnaire sur une échelle de 1 à 5. Les variables sont :
Sleep Quality (qualité du sommeil)
Headache Frequency (fréquence hebdomadaire des maux de tête)
Academic Performance (performance académique auto-évaluée)
Study Load (charge de travail d’étude)
Extracurricular Activities (fréquence des activités extrascolaires)
Stress Level (niveau de stress auto-évalué, cible).
La table extraite dans le notebook affiche notamment que le DataFrame initial contient 520 lignes sur 6 colonnes (ex. : la première ligne affiche SleepQuality=3, Headache=1, PerfAcad=3, StudyLoad=4, Extra=2, Stress=3). Ces données proviennent vraisemblablement d’un ensemble de données « Student Stress Factors » (source originale non précisée dans le notebook, mais souvent issu de travaux en intelligence artificielle sur le stress étudiant). La préparation des données comprend le chargement du fichier CSV dans un DataFrame Pandas, le contrôle des doublons et des valeurs manquantes, la vérification des types et une première description statistique (moyenne, écart-type, etc.) via df.describe(). 
On constate qu’aucune valeur manquante n’est présente. Le notebook supprime les doublons, réduisant le jeu de 520 à 104 observations uniques, ce qui sera la base de l’analyse ultérieure.

## Exploration des données et prétraitement
Après chargement, le DataFrame est examiné avec df.shape, df.info() et df.describe(). Aucune valeur manquante n’est détectée. Le code supprime ensuite les duplicata (df.drop_duplicates()), passant de 520 à 104 étudiants uniques. Le notebook vérifie la distribution de la variable cible (stress) par un diagramme en barres (sns.countplot(x='StressLevel', data=df)), montrant la répartition du niveau de stress parmi les étudiants (bien que l’image ne soit pas intégrée ici, on observe que la cible est décrite comme discrète 1–5). 

Le coefficient de corrélation de Pearson est ensuite calculé pour toutes les paires de variables (df.corr()) et visualisé sous forme de carte de chaleur (heatmap). Cela permet de détecter les relations linéaires. Par exemple, on notera une corrélation positive notable entre Study Load et Stress Level. L’histogramme de corrélation (carte de chaleur) met en évidence que l’axe de charge de travail est corrélé au stress, suggérant que plus la charge d’étude est élevée, plus le stress perçu l’est aussi. Ce type d’analyse exploratoire permet de visualiser rapidement les liens potentiels entre variables avant modélisation.

## Ingénierie des variables
Le notebook crée une nouvelle variable Academic_Pressure_Index, définie comme le produit des scores de « performance académique » et de « charge d’étude » :
```python
df_new['Academic_Pressure_Index'] = df_new['AcademicPerformance'] * df_new['StudyLoad']
```
Cette nouvelle caractéristique vise à synthétiser l’effet combiné de la performance académique et de la charge de travail sur le stress. Aucune justification explicite n’est donnée dans le notebook, mais l’idée est vraisemblablement de capturer l’« indice de pression académique ». Un exemplaire des données agrémenté montre l’apparition de cette colonne en fin de table. Ensuite, toutes les variables qualitatives (mais codées numériquement) sont encodées avec un LabelEncoder de scikit-learn, transformant les valeurs ordinales 1–5 en un encodage 0–4 correspondant (ex. SleepQuality=3 devient 2, etc.). Cette étape normalise les catégories pour la plupart des modèles (bien qu’elles étaient déjà numériques, l’encodage automatique de LabelEncoder décale simplement l’intervalle). 
Le jeu final de données (df_new) comporte donc 104 observations et 7 colonnes (les 5 caractéristiques d’origine encodées, la nouvelle Academic_Pressure_Index, et la cible StressLevel).

## Séparation des jeux d’entraînement et de test
Les variables explicatives sont séparées de la cible : X contient toutes les colonnes sauf StressLevel, et y contient uniquement StressLevel. Le notebook utilise un fractionnement en 75% entraînement / 25% test via train_test_split de sklearn. Cette étape standard prépare les données pour la phase de modélisation. Le commentaire « Target Distribution Imbalance Handling » suggère une préoccupation sur l’équilibre des classes, mais le code se contente de calculer len(X) (104) sans appliquer de technique de rééchantillonnage.

## Modèles statistiques et apprentissage automatique
Le notebook évalue plusieurs modèles de classification supervisée avec leurs performances sur l’ensemble de test. Pour chaque algorithme, le code entraîne le modèle puis affiche un rapport de classification. Les principales étapes sont :
- Régression logistique (Logistic Regression) : algorithme paramétrique de classification binaire multiclasses étendu. La régression logistique estime la probabilité que la cible appartienne à chaque classe grâce à la fonction sigmoïde sur une combinaison linéaire des variables. 
Avec les données fournies, le notebook obtient une précision d’environ 27% sur le test, ce qui est faible (probablement en raison du faible nombre d’échantillons et de la difficulté intrinsèque).

- Arbre de décision (Decision Tree) : modèle supervisé non paramétrique qui partitionne récursivement l’espace des données en fonction de tests sur les caractéristiques. Ici, le DecisionTreeClassifier par défaut est utilisé. La précision rapportée sur le test n’est que de ~15%, indiquant un mauvais ajustement (le modèle semble sur-ajusté malgré la petite taille de données).

- K plus proches voisins (k-Nearest Neighbors) : algorithme non paramétrique attribuant la classe majoritaire parmi les k voisins les plus proches (selon une métrique) d’un point à classer. Le notebook utilise KNeighborsClassifier avec k=5 (par défaut). La précision obtenue est d’environ 23%. Le faible score peut être dû à la dimensionnalité limitée et à la structure des données.

- Naïf Bayésien Gaussien (Gaussian Naive Bayes) : classifieur probabiliste supposant l’indépendance conditionnelle des variables explicatives et une distribution gaussienne pour chaque variable continue. Ce modèle obtient ici la meilleure précision parmi les modèles simples (~31%). Cela peut s’expliquer par sa robustesse sur petits jeux de données et sa capacité à générer des probabilités même lorsque les données sont continues.

- Naïf Bayésien Multinomial (Multinomial Naive Bayes) : variante du Naïf Bayes adaptée aux données sous distribution multinomiale (fréquences/distributions de comptage). Bien que les variables du dataset ne soient pas des comptages, ce modèle est testé pour comparaison. La précision chute à ~19%.

- Machine à vecteurs de support (Support Vector Classifier – SVC) : algorithme qui cherche un hyperplan séparant les classes avec la marge la plus grande. En pratique, SVC par défaut (kernel RBF) est utilisé. La performance est autour de 23%. Le faible score suggère que les données ne sont pas linéairement séparables et que le kernel choisi n’améliore pas suffisamment la classification.

- Forêt aléatoire (Random Forest) : ensemble d’arbres de décision entraînés sur des échantillons bootstrapés et sous-ensembles aléatoires de caractéristiques. L’idée est de réduire le surapprentissage d’un arbre unique en moyennant plusieurs arbres. Le modèle RandomForestClassifier donne une précision d’environ 23% sur le test. Ce score reste modeste, bien que la validation croisée ultérieure lui attribue le meilleur résultat global parmi les modèles testés.

- XGBoost (Extreme Gradient Boosting) : méthode de boosting séquentiel optimisée, basée sur la combinaison d’arbres de décision faibles où chaque nouvel arbre corrige les erreurs des précédents. XGBClassifier est utilisé. Le score sur le test est très faible (~19%), indiquant peut-être un sur-apprentissage ou l’inadaptation aux données sans réglage des hyperparamètres.

- Perceptron multicouche (Multi-layer Perceptron, MLP) : réseau de neurones à propagation avant avec une ou plusieurs couches cachées. MLPClassifier par défaut est appliqué. La précision observée est ~27%, comparable à la régression logistique. Cela suggère que le réseau n’est pas parvenu à extraire des relations non linéaires significatives avec si peu de données.

- Boosting Gradient : implémentation standard du gradient boosting (GradientBoostingClassifier) combinant séquentiellement des arbres faiblement profonds. La précision sur le test est également d’environ 27%.

Dans l’ensemble, les performances des modèles individuels sont faibles (de 15% à 31%). Un résumé des précisions sur le jeu de test est le suivant :
Logistic Regression : ~27%
Decision Tree : ~15%
KNN : ~23%
Gaussian NB : ~31% (meilleur parmi ceux-ci)
Multinomial NB : ~19%
SVC : ~23%
Random Forest : ~23%
XGBoost : ~19%
MLP : ~27%
Gradient Boost : ~27%

Ces résultats montrent que les modèles testés éprouvent des difficultés pour prédire le niveau de stress avec le jeu de données fourni.
### Validation croisée
Pour évaluer plus robustement les performances, le notebook applique une validation croisée stratifiée K-Fold (K=10) sur chaque modèle. La validation croisée (K-Fold) consiste à diviser les données en K sous-ensembles, à entraîner K modèles sur K-1 plis et à tester sur le pli restant à chaque itération. Cette méthode permet d’estimer la performance moyenne du modèle sur différentes partitions et de réduire la variance due à un seul découpage. Le code affiche la précision moyenne (accuracy) pour chaque modèle. Les résultats moyens sont :
Gaussian Naive Bayes : ~37.5% (meilleure précision)
Decision Tree : ~24.0% (plus faible précision)
Random Forest : ~34.5%
KNN : ~31.5%
MLP : ~34.0%
Gradient Boost : ~32.5%
autres modèles entre ~28% et 31%.

Ainsi, la validation croisée confirme que le classifieur Naïf Bayésien Gaussien est le plus performant (≈37%) tandis que l’arbre de décision est le moins performant (≈24%). Ces valeurs très modestes indiquent que, sans amélioration des données ou des méthodes (p. ex. réglage des hyperparamètres, ajout de nouvelles variables), les modèles ne prédisent pas bien le stress.

## Conclusion
Les résultats du code proposent une exploration rigoureuse du jeu de données sur le stress étudiant et compare de nombreux algorithmes classiques de machine learning. L’analyse confirme que la charge d’étude semble corrélée au stress perçu (ce qui cadre avec l’intuition que plus la charge est forte, plus le stress augmente). Toutefois, la prédiction précise du stress reste difficile : les meilleures approches (Gaussian NB, Random Forest) n’atteignent qu’une précision moyenne d’environ 34–37%. Ces scores faibles s’expliquent en partie par le jeu de données réduit après retrait des doublons (104 étudiants) et la nature auto-rapportée des indicateurs. 

Les limites de l’étude sont donc le faible nombre d’exemples exploités et l’absence de validation externe plus large. Pour améliorer ce travail, il faudrait collecter davantage de données ou enrichir le jeu de variables (par exemple inclure des mesures objectives de stress ou d’autres facteurs psycho-sociaux). Il serait également utile d’optimiser les hyperparamètres (grid search) et d’explorer d’autres modèles (forêts de gradient plus complexes, méthodes à base de réseaux de neurones profonds, ou techniques de sur-échantillonnage pour corriger d’éventuels déséquilibres). Enfin, l’utilisation de méthodes explicatives (p. ex. SHAP) pourrait clarifier l’influence relative de chaque facteur de risque de stress, comme cela a été fait dans d’autres études scientifiques. 

En somme, ce rapport détaillé montre toutes les étapes du notebook – de l’importation des bibliothèques (Pandas pour la manipulation des données, NumPy pour les calculs numériques, Matplotlib/Seaborn pour la visualisation) à la modélisation avec sklearn – en contextualisant chaque choix par rapport aux objectifs de prédiction du stress étudiant.
