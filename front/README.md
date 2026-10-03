# Front - Interface Gradio

Interface web intuitive pour explorer les données du projet de prédiction énergétique.

## 🚀 Installation

```powershell
cd front
pip install gradio plotly requests
```

## 📖 Utilisation

### Lancer l'interface

```powershell
python app.py
```

L'interface sera accessible sur `http://localhost:7860`

### Fonctionnalités

L'interface propose 4 onglets :

1. **📋 Vue d'ensemble**
   - Informations générales sur les datasets (household, appliances)
   - Métriques : nombre de features, lignes train/val/test, périodes
   - Variable cible

2. **🔍 Exploration des données**
   - Visualisation des données brutes
   - Choix du dataset (household/appliances)
   - Choix du split (train/val/test)
   - Nombre de lignes à afficher (5-50)
   - Graphique temporel de la variable cible

3. **🔧 Features**
   - Liste complète des features disponibles
   - Variable cible
   - Nombre total de features

4. **🔌 API Status**
   - Vérification de l'état de l'API
   - Liste des endpoints disponibles

## 🔧 Dépendances

- gradio>=5.16.0
- plotly>=5.24.1
- requests
- pandas

## 📝 Note

L'interface nécessite que l'API FastAPI soit lancée sur `http://localhost:8000`. Voir `../README.md` pour lancer l'API.
