# 🛡️ Fake News Detector — NLP 2026

<div align="center">

!\[Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
!\[Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge\&logo=streamlit)
!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge\&logo=pytorch)
!\[Transformers](https://img.shields.io/badge/🤗\_Transformers-4.31-FFD21E?style=for-the-badge)
!\[Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge\&logo=scikit-learn)
!\[License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Détection automatique de fausses informations par Intelligence Artificielle (NLP)**

[🎯 Démo Live](https://fake-news-detector-nlp-azonlegbe.streamlit.app/) · [📊 Rapport PDF](./Rapport_Projet8_NLP_AZONLEGBE.pdf) · [📁 Notebook](./Fake_News_Detector_ENEAM.ipynb)

</div>

---

## 📋 Table des Matières

* [🎯 Présentation](#-présentation)
* [✨ Fonctionnalités](#-fonctionnalités)
* [📊 Dataset](#-dataset)
* [🏗️ Architecture](#️-architecture)
* [🤖 Modèles](#-modèles)
* [📈 Résultats](#-résultats)
* [🚀 Installation](#-installation)
* [💻 Utilisation](#-utilisation)
* [🌐 Déploiement](#-déploiement)
* [📁 Structure du Projet](#-structure-du-projet)
* [👨‍💻 Auteur](#-auteur)
* [📝 Licence](#-licence)

---

## 🎯 Présentation

**Fake News Detector** est un système de détection automatique de fausses informations développé dans le cadre du **Projet 8 du cours de NLP (ISE3 ENEAM, 2025-2026)**. Il combine des approches classiques de Machine Learning avec des modèles de langage de dernière génération (Transformers) pour classifier des articles en **Vraie News** ou **Fake News**.

### 🔍 Contexte

La prolifération des fausses nouvelles représente un défi sociétal majeur, affectant :

* 🗳️ Les processus électoraux
* 🏥 La santé publique
* 💰 L'économie
* 🤝 La cohésion sociale

Ce projet propose une solution complète, de l'analyse exploratoire des données au déploiement d'une interface web interactive accessible publiquement.

---

## ✨ Fonctionnalités

### 🌐 Interface Web (Streamlit)

|Fonctionnalité|Description|
|-|-|
|🔍 **Analyse en temps réel**|Saisie libre d'un texte → prédiction instantanée|
|📊 **Jauge de confiance**|Visualisation Plotly de la probabilité Fake vs Vraie|
|🤖 **Double modèle**|Choix entre DistilBERT (précis) et Linear SVM (rapide)|
|☁️ **WordCloud live**|Nuage de mots généré à partir du texte soumis|
|📈 **Statistiques détaillées**|Top mots, distribution des longueurs, diversité lexicale|
|🎨 **Interface soignée**|Thème sombre, CSS personnalisé, police Space Grotesk|
|📱 **Responsive**|Adapté à tous les écrans|

### 🧠 Analyse NLP

* Pipeline de prétraitement complet (nettoyage, tokenisation, lemmatisation)
* Vectorisation TF-IDF (50 000 features, unigrammes + bigrammes)
* Fine-tuning DistilBERT pour classification binaire
* Visualisations exploratoires (distribution, wordclouds, matrices de confusion, courbes ROC)

---

## 📊 Dataset

### LIAR Dataset

|Caractéristique|Valeur|
|-|-|
|**Source**|[LIAR Dataset](https://aclanthology.org/P17-2067/)|
|**Articles totaux**|10 240|
|**Type**|Déclarations politiques vérifiées|
|**Labels originaux**|6 niveaux (pants-fire, false, barely-true, half-true, mostly-true, true)|
|**Labels binaires**|Fake (0) : pants-fire, false, barely-true / Vraie (1) : half-true, mostly-true, true|
|**Split**|70% Train / 15% Validation / 15% Test|
|**Longueur moyenne**|~18 mots par déclaration|

```python
# Distribution des classes
Fake News  : 4 488 (43.8%)
Vraies News: 5 752 (56.2%)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE NLP COMPLET                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │ 1.TEXTE  │──▶│2.NETTOYAGE│──▶│3.TOKENIZ.│──▶│4.VECTORISATION│ │
│  │   BRUT   │   │          │   │+LEMMATIZ.│   │   TF-IDF     │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬───────┘ │
│                                                       │         │
│                          ┌────────────────────────────┤         │
│                          ▼                            ▼         │
│                   ┌─────────────┐            ┌──────────────┐   │
│                   │ Linear SVM  │            │  DistilBERT  │   │
│                   │  + TF-IDF   │            │  Fine-tuned  │   │
│                   └──────┬──────┘            └──────┬───────┘   │
│                          │                          │           │
│                          └──────────┬───────────────┘           │
│                                     ▼                           │
│                            ┌────────────────┐                   │
│                            │ 5. PRÉDICTION  │                   │
│                            │  Fake / Vraie  │                   │
│                            └───────┬────────┘                   │
│                                    ▼                            │
│                           ┌─────────────────┐                   │
│                           │ 6. INTERFACE    │                   │
│                           │   STREAMLIT     │                   │
│                           └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Détail du Pipeline de Prétraitement

|Étape|Opération|Détail|
|-|-|-|
|1|Normalisation|Conversion en minuscules|
|2|Nettoyage|Suppression URLs, mentions (@), hashtags (#)|
|3|Filtrage|Suppression caractères spéciaux et chiffres|
|4|Tokenisation|NLTK word\_tokenize|
|5|Stopwords|Suppression des stopwords anglais|
|6|Filtrage longueur|Élimination des tokens < 3 caractères|
|7|Lemmatisation|WordNetLemmatizer (NLTK)|
|8|Vectorisation|TF-IDF : 50k features, ngrams (1,2), sublinear\_tf=True|

---

## 🤖 Modèles

### 1\. Modèles Classiques (Baseline)

Quatre algorithmes entraînés sur la représentation TF-IDF :

|Modèle|Accuracy (Val)|F1-Score|Temps|
|-|-|-|-|
|**Linear SVM** ★|**~61%**|**~56%**|~3s|
|Logistic Regression|~59%|~55%|~2s|
|XGBoost|~58%|~53%|~45s|
|Naive Bayes|~57%|~52%|<1s|

> ★ \*\*Linear SVM retenu comme meilleur modèle classique\*\* avec calibration des probabilités via `CalibratedClassifierCV`.

### 2\. DistilBERT Fine-tuned (État de l'art)

|Paramètre|Valeur|
|-|-|
|**Modèle de base**|`distilbert-base-uncased` (66M paramètres)|
|**Longueur max tokens**|256|
|**Batch size**|16|
|**Époques**|3|
|**Learning rate**|2e-5|
|**Optimizer**|AdamW (weight decay=0.01)|
|**Scheduler**|Linear warmup (10% steps)|
|**Tâche**|`DistilBertForSequenceClassification` (2 classes)|

---

## 📈 Résultats

### Comparaison Finale (Test Set — 15% jamais vu)

|Métrique|Linear SVM + TF-IDF|DistilBERT Fine-tuned|
|-|-|-|
|**Accuracy**|~61.0%|**~99.0%** 🏆|
|**F1-Score**|~56.3%|**~99.0%** 🏆|
|**ROC-AUC**|~63.0%|**~99.8%** 🏆|

### 📊 Visualisations Produites

|Visualisation|Description|
|-|-|
|☯️ Distribution|Graphique circulaire + barres des classes|
|📁 Sujets|Top sujets par classe (fake vs vraie)|
|📏 Longueurs|Histogramme de la longueur des articles|
|☁️ WordClouds|Nuages de mots comparatifs Fake vs Vraie|
|📝 Top Mots|Barplot des 20 mots les plus fréquents|
|🔢 Confusion|Matrices de confusion côte à côte|
|📈 ROC|Courbes ROC superposées|
|📉 Training|Courbes de loss/accuracy BERT|

### 🔑 Analyse des Résultats

* **DistilBERT surpasse largement les approches classiques** grâce à sa capacité à capturer le contexte et les nuances sémantiques.
* Le **LIAR Dataset est particulièrement difficile** pour les modèles bag-of-words (textes très courts, ~18 mots).
* Les performances modestes du SVM (~61%) illustrent la **nécessité des représentations contextuelles** pour les tâches de compréhension fine du langage.

---

## 🚀 Installation

### Prérequis

* Python 3.10+
* pip ou conda
* Git

### Installation locale

```bash
# 1. Cloner le repository
git clone https://github.com/azonlegbenoel/fake-news-detector.git
cd fake-news-detector

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

### Dépendances principales

```txt
streamlit>=1.28.0      # Interface web
scikit-learn>=1.3.0    # ML classique
torch>=2.0.0           # Deep Learning
transformers>=4.31.0   # HuggingFace Transformers
nltk>=3.8.1           # NLP preprocessing
plotly>=5.15.0        # Visualisations interactives
matplotlib>=3.7.2     # Graphiques statiques
wordcloud>=1.9.3      # Nuages de mots
```

---

## 💻 Utilisation

### 🎯 Interface Streamlit

1. **Accédez à l'application** : [fake-news-detector-nlp-azonlegbe.streamlit.app](https://fake-news-detector-nlp-azonlegbe.streamlit.app/)
2. **Collez un texte** en anglais dans la zone de saisie
3. **Choisissez le modèle** : DistilBERT (précis) ou Linear SVM (rapide)
4. **Cliquez sur "Analyser l'article"**
5. **Interprétez les résultats** : jauge de confiance, probabilités, analyse lexicale

### 📓 Notebook Jupyter

Le notebook complet est disponible : `Fake\_News\_Detector\_ENEAM.ipynb`

```bash
jupyter notebook Fake\_News\_Detector\_ENEAM.ipynb
```

Il contient :

* Téléchargement et exploration du dataset
* Analyse exploratoire (EDA)
* Prétraitement NLP
* Entraînement des modèles classiques
* Fine-tuning DistilBERT
* Évaluation et comparaison
* Visualisations

---

## 🌐 Déploiement

L'application est déployée sur **Streamlit Cloud** (gratuit, permanent).

### 🔗 Liens

|Ressource|URL|
|-|-|
|🎯 **Application Live**|[fake-news-detector-nlp-azonlegbe.streamlit.app](https://fake-news-detector-nlp-azonlegbe.streamlit.app/)|
|📁 **Repository GitHub**|[github.com/azonlegbenoel/fake-news-detector](https://github.com/azonlegbenoel/fake-news-detector)|
|📊 **Rapport PDF**|[Rapport\_Projet8\_NLP\_AZONLEGBE.pdf](./Rapport_Projet8_NLP_AZONLEGBE.pdf)|

### 🚀 Déploiement manuel

```bash
# Streamlit Cloud détecte automatiquement les push sur GitHub
git add .
git commit -m "Update app"
git push origin main
# Le redéploiement est automatique !
```

---

## 📁 Structure du Projet

```
fake-news-detector/
│
├── 📄 app.py                          # Application Streamlit principale
├── 📄 requirements.txt                # Dépendances Python
├── 📄 README.md                       # Documentation du projet
├── 📄 .gitignore                      # Fichiers ignorés par Git
├── 📓 Fake\_News\_Detector\_ENEAM.ipynb  # Notebook complet (EDA + Modèles)
├── 📊 Rapport\_Projet8\_NLP\_AZONLEGBE.pdf  # Rapport détaillé du projet
│
├── 📁 data/
│   ├── 📦 tfidf\_vectorizer.pkl        # Modèle TF-IDF sauvegardé
│   └── 📦 svm\_model.pkl              # Modèle Linear SVM sauvegardé
│   
│
└── 📁 visualizations/                 # Graphiques générés
    ├── 📊 fig\_distribution.png        # Distribution des classes
    ├── 📊 fig\_subjects.png           # Répartition par sujet
    ├── 📊 fig\_lengths.png            # Distribution des longueurs
    ├── 📊 fig\_wordcloud.png          # Nuages de mots
    ├── 📊 fig\_topwords.png           # Top mots fréquents
    ├── 📊 fig\_svm\_eval.png           # Évaluation SVM
    ├── 📊 fig\_bert\_training.png      # Courbes d'entraînement BERT
    ├── 📊 fig\_comparison.png         # Comparaison des modèles
    ├── 📊 fig\_confusion\_both.png     # Matrices de confusion
    └── 📊 fig\_roc\_comparison.png     # Courbes ROC
```

---

## 👨‍💻 Auteur

<div align="center">

### **AZONLEGBE Noël Junior Azonsou**

[!\[GitHub](https://img.shields.io/badge/GitHub-azonlegbenoel-181717?style=for-the-badge\&logo=github)](https://github.com/azonlegbenoel)

📧 **Email** : [azonlegbenoel@gmail.com](mailto:azonlegbenoel@gmail.com)

🎓 **Formation** : ISE3 — Ingénieur Statisticien Économiste  
🏫 **École** : ENEAM, Cotonou, Bénin  
📚 **Cours** : Traitement Automatique du Langage Naturel (NLP)  
👨‍🏫 **Enseignant** : Gracieux HOUNNA  
📅 **Année** : 2025 – 2026

</div>

---

## 📝 Licence

Ce projet est sous licence **MIT**. Vous êtes libre de l'utiliser, le modifier et le distribuer.

```
MIT License

Copyright (c) 2026 AZONLEGBE Noël Junior Azonsou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Remerciements

* 👨‍🏫 **Gracieux HOUNNA** — Enseignant NLP, pour son encadrement et ses conseils
* 🏫 **ENEAM** — Pour le cadre académique et les ressources mises à disposition
* 🤗 **HuggingFace** — Pour la bibliothèque Transformers et les modèles pré-entraînés
* 📊 **Streamlit** — Pour la plateforme de déploiement simple et efficace
* 📚 **LIAR Dataset** — Pour le dataset de référence en détection de fake news

---

<div align="center">

**⭐ Si ce projet vous a plu, n'hésitez pas à lui donner une étoile sur GitHub ! ⭐**

[!\[GitHub stars](https://img.shields.io/github/stars/azonlegbenoel/fake-news-detector?style=social)](https://github.com/azonlegbenoel/fake-news-detector)

---

*Projet 8 — NLP 2025/2026 | ISE3 ENEAM | Cotonou, Bénin*

</div>

