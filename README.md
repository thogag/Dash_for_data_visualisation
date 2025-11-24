# AD_projet



## Lancement de l'application DASH

Pour lancer l'application Dash, suivez les étapes ci-dessous :

1.  Ouvrez un terminal dans le dossier racine du projet (là où se trouve
    `run.sh`).

2.  Assurez-vous que le script possède les droits d'exécution :

    ``` bash
    chmod +x run.sh
    ```

3.  Exécutez le script de lancement :

    ``` bash
    ./run.sh
    ```

4.  Le script va automatiquement :

    -   créer un environnement virtuel `venv/`
    -   installer toutes les dépendances depuis `requirements.txt`
    -   lancer l'application Dash via `app.py`

Une fois l'application démarrée, ouvrez un navigateur et accédez à
l'adresse affichée dans le terminal, généralement :

    http://127.0.0.1:8050

## Présentation rapide des onglets de l'application

L’application Dash propose plusieurs pages permettant d’explorer et d’analyser le dataset des sports ESPN :

### Accueil
Page centrale donnant accès à toutes les analyses du projet via des boutons thématiques.

### YData Profiling
Affiche le rapport automatique généré par YData : statistiques descriptives, distributions, corrélations et qualité des données.

### Régression Linéaire
Permet de choisir une variable dépendante et plusieurs variables explicatives.  
Affiche les points observés et les droites de régression correspondantes.

### PCA
Projette les sports dans un espace 2D selon les composantes principales.  
Affiche la variance expliquée et les caractéristiques les plus contributives aux axes.

### Clusters
Sélection de deux variables pour réaliser un clustering K-Means.  
Affiche le scatter plot ainsi que le Silhouette Plot pour évaluer la qualité du clustering.

### Corrélations
Heatmap des corrélations entre les variables numériques pour repérer les relations fortes ou redondantes.

### Rapport
Ouvre le rapport rédigé présentant le dataset, la méthodologie, les analyses et les principales conclusions.

Le dashboard regroupe l’ensemble des analyses réalisées dans le cadre du projet.

## Auteurs 

Matthieu Guigard -- Thomas Gagnieu -- Séverin Jorry