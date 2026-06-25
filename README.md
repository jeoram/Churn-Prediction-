# churn-mutuelle

Projet de démonstration pour l’analyse de churn en assurance mutuelle.

## Structure

- data/ : données brutes ou générées localement
- src/ : modules Python modulaires
- outputs/ : graphiques, métriques et artefacts générés
- main.py : point d’entrée du pipeline

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution

```bash
python main.py
```

Le pipeline utilise par défaut un jeu de données public de churn réel si le téléchargement est possible, sinon il bascule sur un dataset synthétique. Il exécute ensuite :
- EDA descriptif (histogrammes, boxplots, heatmap)
- Feature engineering
- Entraînement de plusieurs modèles
- Évaluation avec métriques et graphiques
