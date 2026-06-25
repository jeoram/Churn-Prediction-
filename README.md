# Churn Prediction - Projet de prédiction de résiliation

Ce projet présente un pipeline complet de machine learning pour prédire la résiliation client (churn) à partir d’un jeu de données réel de télécommunication, adapté à un contexte assurance mutuelle.

## Objectif

Le but est de montrer une démarche de data science complète :
- exploration et visualisation des données
- préparation des features
- comparaison de plusieurs modèles de classification
- évaluation avec des métriques business-relevant

## Ce que contient le projet

- un jeu de données réel utilisé par défaut
- un pipeline Python modulaire
- des visualisations EDA et de performance
- plusieurs modèles de classification :
  - Régression logistique
  - Random Forest
  - XGBoost

## Structure du projet

- data/ : données utilisées par le projet
- src/ : modules Python organisés par fonctionnalité
  - data_loader.py : chargement et préparation des données
  - eda.py : analyses descriptives et graphiques
  - preprocessing.py : feature engineering et préparation des données
  - modeling.py : entraînement des modèles
  - evaluation.py : métriques et visualisations de performance
- outputs/ : graphiques, CSV de métriques et résultats générés
- main.py : point d’entrée du pipeline

## Prérequis

- Python 3.10+
- pip

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

Le script génère automatiquement :
- des graphiques EDA
- des courbes ROC et Precision-Recall
- des matrices de confusion
- un fichier CSV récapitulatif des métriques
- une heatmap de comparaison des modèles

## Résultats attendus

Le pipeline affiche un résumé des performances des modèles selon les métriques suivantes :
- Accuracy
- AUC-ROC
- PR-AUC
- F1-score

## Dépendances principales

Les bibliothèques utilisées sont :
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost

## Auteur

Projet réalisé dans un cadre de démonstration de machine learning et data visualization.
