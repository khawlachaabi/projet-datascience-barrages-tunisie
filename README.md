# Dashboard d'analyse des barrages en Tunisie 💧

Dashboard interactif développé avec Streamlit pour analyser les données des barrages en Tunisie par région : taux de remplissage, apports, lâchers, évolution temporelle, comparaisons et carte géographique.

## ✨ Fonctionnalités

- **Filtres** :
  - Sélection des régions
  - Filtrage optionnel par plage de dates
- **Onglets** :
  - **Vue d'ensemble** : KPIs globaux, courbes stock / taux de remplissage, répartition par région
  - **Analyse par région** : indicateurs détaillés, évolution, bilan hydrique, analyse saisonnière
  - **Comparaison** : comparaison multi-régions (taux, apports, lâchers, radar multi-critères)
  - **Données brutes** : tableau filtrable + export CSV
  - **Map** : carte des barrages (Mapbox) avec taux de remplissage
- **Export des données** :
  - CSV (plusieurs fichiers)
  - Excel (plusieurs feuilles : données brutes, stats par région, stats par barrage)
- **Sidebar avancée** :
  - Aide intégrée
  - Statistiques rapides (meilleure / pire région)
  - Recherche de barrage par nom

## 🛠️ Technologies

- Python
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- Plotly (express, graph_objects, make_subplots)
- Fichiers de données :
  - `Barrages_tn.csv`
  - `barrages.csv`

Fichier principal : `dashboard.py`

## 🚀 Installation & exécution

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/khawlachaabi/projet-barrages-datascience.git
   cd "Projet datascience"
   ```
2. Installer les dépendances :
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly openpyxl
   ```
3. Lancer le dashboard :
   ```bash
   streamlit run dashboard.py
   ```

Le navigateur s'ouvrira automatiquement sur l'interface.

## 💡 Utilisation

- Utiliser la **sidebar** pour :
  - Choisir les régions
  - Activer / désactiver le filtrage par dates
  - Accéder à l'aide, à la recherche de barrage, à l'export avancé, etc.
- Naviguer entre les onglets pour explorer les différentes analyses.
- Télécharger les données filtrées au format CSV ou Excel.

## 👤 Auteur

- **Nom** : Khawla Chaabi  
- **GitHub** : https://github.com/khawlachaabi

