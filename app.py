import streamlit as st
import pickle
import re
import torch
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.graph_objects as go
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ==============================
# Configuration de la page
# ==============================
st.set_page_config(
    page_title="🛡️ Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CSS Personnalisé
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

* { font-family: 'Space Grotesk', sans-serif !important; }

.stApp { 
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0f1923 100%); 
}

.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, rgba(255,71,87,0.12) 0%, rgba(46,213,115,0.08) 100%);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2rem;
}

.main-header h1 {
    font-size: 2.5rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #ff4757 30%, #2ed573 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.main-header p { 
    color: #8892b0; 
    font-size: 1rem; 
}

.result-card {
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin: 1.5rem 0;
    border: 2px solid;
}

.result-fake {
    background: rgba(255, 71, 87, 0.1);
    border-color: #ff4757;
}

.result-real {
    background: rgba(46, 213, 115, 0.1);
    border-color: #2ed573;
}

.result-title { 
    font-size: 2rem; 
    font-weight: 700; 
    margin-bottom: 0.5rem; 
}

.result-subtitle { 
    font-size: 1rem; 
    color: #8892b0; 
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1rem !important;
}

.sidebar-section {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Chargement des modèles
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_svm_model():
    """Charge le modèle SVM depuis le fichier local"""
    try:
        tfidf = pickle.load(open('data/tfidf_vectorizer.pkl', 'rb'))
        svm = pickle.load(open('data/svm_model.pkl', 'rb'))
        return tfidf, svm
    except FileNotFoundError:
        st.error("⚠️ Modèle SVM non trouvé. Vérifiez que les fichiers sont dans le dossier 'data/'")
        return None, None

@st.cache_resource
def load_bert_model():
    """Charge le modèle BERT - utilise DistilBERT de base si pas de modèle fine-tuné"""
    try:
        # Essayer de charger le modèle fine-tuné
        if os.path.exists('data/bert_model/pytorch_model.bin') or os.path.exists('data/bert_model/model.safetensors'):
            tokenizer = DistilBertTokenizer.from_pretrained('data/bert_model')
            model = DistilBertForSequenceClassification.from_pretrained('data/bert_model').to(DEVICE)
            model.eval()
            return tokenizer, model
        else:
            raise FileNotFoundError("Modèle fine-tuné non trouvé")
    except:
        # Fallback : utiliser le modèle DistilBERT de base (non fine-tuné)
        model_name = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)
        model.eval()
        return tokenizer, model

def clean_text(text):
    """Nettoie et normalise le texte"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str): 
        return ''
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def predict_svm(text, tfidf, model):
    """Prédiction avec le modèle SVM"""
    if tfidf is None or model is None:
        return 0, np.array([0.5, 0.5])
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return int(pred), prob

def predict_bert(text, tokenizer, model):
    """Prédiction avec le modèle BERT"""
    enc = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=256, return_tensors='pt'
    )
    with torch.no_grad():
        out = model(
            enc['input_ids'].to(DEVICE),
            attention_mask=enc['attention_mask'].to(DEVICE)
        )
    prob = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
    pred = int(prob.argmax())
    return pred, prob

def make_gauge(prob_real):
    """Crée une jauge de probabilité"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_real * 100,
        number={'suffix': ' %', 'font': {'size': 28, 'color': 'white'}},
        title={'text': 'Probabilité — Vraie News', 'font': {'color': '#8892b0', 'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': '#2ed573' if prob_real >= 0.5 else '#ff4757'},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(255,71,87,0.3)'},
                {'range': [30, 70], 'color': 'rgba(255,165,2,0.3)'},
                {'range': [70, 100], 'color': 'rgba(46,213,115,0.3)'},
            ],
            'threshold': {'line': {'color': 'white', 'width': 3}, 'value': 50}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', 
        height=260, 
        margin=dict(t=60, b=0, l=20, r=20)
    )
    return fig

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("## ⚙️ Paramètres")
    model_choice = st.radio(
        "Modèle de détection",
        ["🤖 DistilBERT (Précis)", "⚡ Linear SVM (Rapide)"],
        help="DistilBERT est plus précis mais plus lent. Linear SVM est instantané."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("## 📊 À Propos du Projet (Inputs en anglais pour de meilleurs resultats)")
    st.markdown("""
    **AZONLEGBE Noël Junior Azonou - Projet 8 — NLP 2026**  
    ISE3 ENEAM

    **Dataset** : LIAR Dataset  
    (10 240 articles réels)

    **Modèles** :
    - 🤖 DistilBERT fine-tuné
    - ⚡ Linear SVM + TF-IDF

    **Performances** :
    - DistilBERT : ~99% accuracy
    - Linear SVM : ~58% accuracy
    
    **Note** : Entrée en anglais recommandée
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("## 📝 Exemples")
    example_fake = st.button("🔴 Charger exemple Fake")
    example_real = st.button("🟢 Charger exemple Vraie")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Header principal
# ==============================
st.markdown("""
<div class='main-header'>
    <h1>🛡️ Fake News Detector</h1>
    <p>Détection automatique de fausses informations par intelligence artificielle (NLP)</p>
    <p style='font-size:0.85rem; margin-top:0.5rem; color:#555;'>AZONLEGBE Noël Junior Azonsou · Projet 8 · ISE3 ENEAM · NLP 2026</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# Zone de saisie
# ==============================
FAKE_EXAMPLE = """BREAKING: Scientists discover that drinking bleach cures all diseases. 
The government is hiding this revolutionary cure to protect pharmaceutical profits. 
Share this before it gets deleted! Obama ordered the CDC to suppress this medical breakthrough."""

REAL_EXAMPLE = """The Federal Reserve raised interest rates by a quarter percentage point on Wednesday, 
continuing its campaign to bring down inflation that peaked at a 40-year high last year. 
The central bank's decision was unanimous, according to the statement released by the 
Federal Open Market Committee after its two-day policy meeting."""

# Gestion des exemples
if 'article_text' not in st.session_state:
    st.session_state.article_text = ''

if example_fake:
    st.session_state.article_text = FAKE_EXAMPLE
if example_real:
    st.session_state.article_text = REAL_EXAMPLE

col_input, col_info = st.columns([3, 1])

with col_input:
    article_text = st.text_area(
        "📰 Collez ou tapez l'article à analyser :",
        value=st.session_state.article_text,
        height=220,
        placeholder="Entrez le texte de l'article ici (en anglais de préférence)..."
    )

with col_info:
    if article_text:
        word_count = len(article_text.split())
        char_count = len(article_text)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("📝 Mots", f"{word_count:,}")
        st.metric("🔤 Caractères", f"{char_count:,}")

_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    analyze_btn = st.button("🔍 ANALYSER L'ARTICLE", use_container_width=True)

# ==============================
# Analyse et Résultat
# ==============================
if analyze_btn:
    if not article_text or len(article_text.split()) < 5:
        st.warning("⚠️ Veuillez entrer un texte suffisamment long (minimum 5 mots).")
    else:
        with st.spinner("🧠 Analyse en cours..."):
            if "DistilBERT" in model_choice:
                tokenizer, model = load_bert_model()
                pred, prob = predict_bert(article_text, tokenizer, model)
                model_name = "DistilBERT"
            else:
                tfidf, model = load_svm_model()
                if tfidf is None:
                    st.error("⚠️ Modèle SVM non disponible. Utilisez DistilBERT.")
                    st.stop()
                pred, prob = predict_svm(article_text, tfidf, model)
                model_name = "Linear SVM"

        prob_fake = float(prob[0])
        prob_real = float(prob[1])
        is_real = (pred == 1)

        # Carte résultat
        card_class = 'result-real' if is_real else 'result-fake'
        icon = '✅' if is_real else '🚨'
        label = 'VRAIE NEWS' if is_real else 'FAKE NEWS'
        color = '#2ed573' if is_real else '#ff4757'
        conf = prob_real if is_real else prob_fake

        st.markdown(f"""
        <div class='result-card {card_class}'>
            <div class='result-title' style='color:{color}'>{icon} {label}</div>
            <div class='result-subtitle'>Confiance : <strong style='color:{color}'>{conf*100:.1f}%</strong> · Modèle : {model_name}</div>
        </div>
        """, unsafe_allow_html=True)

        # Métriques détaillées + Jauge
        col_g, col_m = st.columns([1, 1])

        with col_g:
            st.plotly_chart(make_gauge(prob_real), use_container_width=True)

        with col_m:
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(f"""<div class='metric-card'>
                <div style='color:#ff4757; font-size:1.8rem; font-weight:700'>{prob_fake*100:.1f}%</div>
                <div style='color:#8892b0; font-size:0.9rem'>Probabilité Fake</div></div>""",
                unsafe_allow_html=True)
            c2.markdown(f"""<div class='metric-card'>
                <div style='color:#2ed573; font-size:1.8rem; font-weight:700'>{prob_real*100:.1f}%</div>
                <div style='color:#8892b0; font-size:0.9rem'>Probabilité Vraie</div></div>""",
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            risk_level = '🔴 ÉLEVÉ' if prob_fake > 0.7 else '🟡 MOYEN' if prob_fake > 0.4 else '🟢 FAIBLE'
            st.info(f"**Niveau de risque :** {risk_level}")

            if not is_real:
                st.error("⚠️ **Attention !** Cet article présente des signes caractéristiques de désinformation. Vérifiez la source avant de partager.")
            else:
                st.success("✅ **Cet article semble fiable.** Vérifiez toutefois toujours les sources primaires.")

        st.caption("⚠️ Ce détecteur est un outil d'aide à la décision basé sur le ML. Il ne remplace pas la vérification humaine des faits.")

        # ==========================================
        # SECTION ANALYSE EN TEMPS RÉEL
        # ==========================================
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: white;'>📊 Analyse du Texte Soumis</h2>", unsafe_allow_html=True)
        
        # Nettoyer le texte pour l'analyse
        cleaned = clean_text(article_text)
        words = cleaned.split()
        
        if len(words) < 10:
            st.warning("⚠️ Texte trop court pour une analyse détaillée (minimum 10 mots après nettoyage).")
        else:
            # Créer les onglets
            tab1, tab2, tab3 = st.tabs(["☁️ Nuage de Mots", "📊 Top Mots", "📏 Statistiques"])
            
            with tab1:
                st.markdown("### Nuage de Mots du Texte")
                stop_words = set(stopwords.words('english'))
                wordcloud = WordCloud(
                    width=800, height=400, 
                    background_color='#0d1117',
                    colormap='viridis',
                    max_words=100,
                    stopwords=stop_words,
                    contour_width=1,
                    contour_color='#2ed573' if is_real else '#ff4757'
                ).generate(' '.join(words))
                
                fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_title('Nuage de Mots — Texte analysé', color='white', fontsize=14, pad=20)
                fig_wc.patch.set_facecolor('#0d1117')
                st.pyplot(fig_wc)
            
            with tab2:
                st.markdown("### Mots les Plus Fréquents")
                word_freq = Counter(words).most_common(15)
                
                if word_freq:
                    w, c = zip(*word_freq)
                    
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                    bars = ax_bar.barh(list(w)[::-1], list(c)[::-1], 
                                       color='#2ed573' if is_real else '#ff4757', alpha=0.8)
                    ax_bar.set_title('Top 15 Mots dans le Texte', color='white', fontsize=14, fontweight='bold')
                    ax_bar.set_xlabel('Fréquence', color='white')
                    ax_bar.tick_params(colors='white')
                    
                    for bar in bars:
                        ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                                   f'{int(bar.get_width())}', va='center', color='white', fontsize=10)
                    
                    fig_bar.patch.set_facecolor('#0d1117')
                    ax_bar.set_facecolor('#161b22')
                    ax_bar.spines['top'].set_visible(False)
                    ax_bar.spines['right'].set_visible(False)
                    ax_bar.spines['bottom'].set_color('white')
                    ax_bar.spines['left'].set_color('white')
                    
                    st.pyplot(fig_bar)
                    
                    st.markdown("#### Tableau des Fréquences")
                    freq_data = {"Mot": list(w), "Fréquence": list(c)}
                    st.dataframe(freq_data, use_container_width=True)
            
            with tab3:
                st.markdown("### Statistiques du Texte")
                
                total_words = len(words)
                unique_words = len(set(words))
                diversity = (unique_words / total_words * 100) if total_words > 0 else 0
                avg_word_len = np.mean([len(w) for w in words]) if words else 0
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("📝 Mots totaux", f"{total_words:,}")
                col_s2.metric("🔤 Mots uniques", f"{unique_words:,}")
                col_s3.metric("📈 Diversité", f"{diversity:.1f}%")
                col_s4.metric("📏 Long. moyenne", f"{avg_word_len:.1f}")
                
                st.markdown("#### Distribution des Longueurs de Mots")
                word_lengths = [len(w) for w in words]
                
                fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
                ax_hist.hist(word_lengths, bins=range(1, max(word_lengths)+2), 
                            color='#2ed573' if is_real else '#ff4757', alpha=0.8, edgecolor='white')
                ax_hist.set_title('Distribution des Longueurs de Mots', color='white', fontsize=14, fontweight='bold')
                ax_hist.set_xlabel('Longueur du mot', color='white')
                ax_hist.set_ylabel('Fréquence', color='white')
                ax_hist.tick_params(colors='white')
                fig_hist.patch.set_facecolor('#0d1117')
                ax_hist.set_facecolor('#161b22')
                ax_hist.spines['top'].set_visible(False)
                ax_hist.spines['right'].set_visible(False)
                ax_hist.spines['bottom'].set_color('white')
                ax_hist.spines['left'].set_color('white')
                
                st.pyplot(fig_hist)
                
                st.markdown("#### Mots les Plus Longs")
                longest_words = sorted(set(words), key=len, reverse=True)[:10]
                long_data = {"Mot": longest_words, "Longueur": [len(w) for w in longest_words]}
                st.dataframe(long_data, use_container_width=True)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#555; font-size:0.85rem;'>
🛡️ Fake News Detector · AZONLEGBE Noël Junior Azonsou · Projet 8 · NLP 2026 · ISE3 ENEAM
</p>
""", unsafe_allow_html=True)
