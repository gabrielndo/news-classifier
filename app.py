import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import numpy as np
from nltk.corpus import stopwords

nltk.download('stopwords')

# Configuração da página
st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="wide"
)

# URL da API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Carrega o modelo e vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return model, vectorizer

# Pre-processamento
def preprocess_text(text):
    stop_words = set(stopwords.words('portuguese'))
    text = text.lower()
    text = re.sub(r'[^a-záéíóúâêîôûãõçàü\s]', '', text)
    text = text.strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

model, vectorizer = load_model()

# Sidebar
st.sidebar.image("https://img.icons8.com/emoji/96/newspaper-emoji.png", width=80)
st.sidebar.title("News Classifier")
st.sidebar.markdown("Classificador automático de notícias brasileiras")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegação",
    ["Sobre mim", "Início", "Análise dos Dados", "Sobre o Modelo", "Classificador"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por **Gabriel Nantes**")
st.sidebar.markdown("Teste Técnico — AeC Centro de Contatos")

# ==================== SOBRE MIM ====================
if pagina == "Sobre mim":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("assets/foto.png", width=250)

    with col2:
        st.title("Gabriel Nantes de Oliveira")
        st.markdown("#### Case para Cientista de Dados Jr.")
        st.markdown("""
        Apaixonado por cinema, futebol, saúde e tecnologia.
        Transformo dados em soluções práticas que geram impacto real —
        do problema à produção.
        """)
        st.markdown("""
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/gabrielndo)
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gabriel-nantes-de-oliveira-784ab9156/)
        """)

    st.markdown("---")
    st.markdown("### Stacks")
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, stack in zip([col1, col2, col3, col4, col5],
                          ["Python", "Azure SQL & Big Query", "Machine Learning", "BI", "Data Analysis"]):
        with col:
            st.markdown(f"""
            <div style='text-align:center; padding:15px; background:#1e1e2e; border-radius:10px'>
                <h4>{stack}</h4>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Projetos")
    col1, col2, col3 = st.columns(3)

    projetos = [
        ("Previsao de Falhas", "Detecção de possíveis falhas em equipamentos por analise de vibracao usando Machine Learning."),
        ("Agente de Pendencias", "Agente inteligente para extracao e categorizacao automatica de falhas e pendencias."),
        ("Monitor de Suplementos", "Aplicativo para lembrete e monitoramento de suplementacao diaria.")
    ]

    for col, (titulo, descricao) in zip([col1, col2, col3], projetos):
        with col:
            st.markdown(f"""
            <div style='padding:20px; background:#1e1e2e; border-radius:10px; height:180px'>
                <h4>{titulo}</h4>
                <p>{descricao}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Soft Skills")

    skills = ['Inteligencia\nEmocional', 'Criatividade', 'Raciocinio\nLogico',
              'Empatia', 'Adaptabilidade']
    valores = [90, 85, 92, 88, 87]
    valores += valores[:1]
    N = len(skills)
    angulos = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.plot(angulos, valores, 'o-', linewidth=2, color='#4fa3e0')
    ax.fill(angulos, valores, alpha=0.25, color='#4fa3e0')
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(skills, color='white', size=6)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.grid(color='grey', alpha=0.3)
    ax.spines['polar'].set_color('grey')

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.pyplot(fig)

# ==================== INÍCIO ====================
elif pagina == "Início":
    st.title("News Classifier")
    st.markdown("Classificador automático de notícias brasileiras usando NLP e Machine Learning.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acurácia", "74%")
    with col2:
        st.metric("Notícias treinadas", "128.652")
    with col3:
        st.metric("Categorias", "18")
    with col4:
        st.metric("Algoritmo", "Log. Regression")

    st.markdown("---")
    st.markdown("## Como funciona?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Entrada
        O sistema recebe o **título** de uma notícia em português e passa por 3 etapas:

        1. **Pre-processamento** — limpa o texto removendo stopwords e pontuação
        2. **Vetorização** — transforma o texto em números com TF-IDF
        3. **Classificação** — a Regressão Logística prevê a categoria
        """)
    with col2:
        st.markdown("""
        ### Saída
        O sistema retorna:

        - **Categoria** — qual das 18 categorias a notícia pertence
        - **Acurácia** — o nível de certeza do modelo na predição

        """)

    st.markdown("---")
    st.markdown("## Etapas")
    st.markdown("**EDA → Pré-processamento → TF-IDF → Regressão Logística → FastAPI → Docker**")

# ==================== ANÁLISE DOS DADOS ====================
elif pagina == "Análise dos Dados":
    st.title("Análise dos Dados")
    st.markdown("Insights extraídos durante a análise exploratória do dataset.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de notícias", "167.053")
    with col2:
        st.metric("Categorias únicas", "18")
    with col3:
        st.metric("Período", "2015 - 2017")

    st.markdown("---")
    st.markdown("### Distribuição de Categorias")
    st.markdown("Desbalanceamento severo — `poder` com 22k notícias vs categorias com menos de 200.")
    st.image("assets/distribuicao_categorias.png", use_container_width=True)

    st.markdown("---")
    st.markdown("### Palavras mais frequentes nos títulos")
    st.markdown("Forte presença de termos políticos — coerente com o período de coleta.")
    st.image("assets/wordcloud.png", use_container_width=True)

    st.markdown("---")
    st.markdown("### Distribuição do tamanho dos títulos")
    st.markdown("Títulos curtos e padronizados com média de 10 palavras.")
    st.image("assets/distribuicao_titulos.png", use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Media de palavras", "10")
    with col2:
        st.metric("Minimo", "1")
    with col3:
        st.metric("Maximo", "23")

# ==================== SOBRE O MODELO ====================
elif pagina == "Sobre o Modelo":
    st.title("Sobre o Modelo")
    st.markdown("Regressão Logística com TF-IDF")

    st.markdown("---")
    st.markdown("### Métricas")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acurácia", "74%")
    with col2:
        st.metric("Notícias de treino", "128.652")
    with col3:
        st.metric("Notícias de teste", "32.163")
    with col4:
        st.metric("Categorias", "18")

    st.markdown("---")
    st.markdown("### Desempenho por Categoria")
    st.markdown("Verde = bom (>0.75) | Laranja = regular (>0.50) | Vermelho = fraco (<0.50)")

    dados = {
        "Categoria": ["esporte", "paineldoleitor", "poder", "mundo", "ilustrada",
                      "mercado", "cotidiano", "saopaulo", "tec", "educacao",
                      "turismo", "colunas", "ciencia", "opiniao", "equilibrioesaude",
                      "ilustrissima", "sobretudo", "tv"],
        "F1-Score": [0.91, 0.91, 0.80, 0.83, 0.78,
                     0.76, 0.77, 0.59, 0.54, 0.67,
                     0.50, 0.56, 0.41, 0.28, 0.34,
                     0.44, 0.29, 0.14]
    }

    df_metricas = pd.DataFrame(dados).sort_values("F1-Score", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['green' if f >= 0.75 else 'orange' if f >= 0.50 else 'red'
              for f in df_metricas["F1-Score"]]
    ax.barh(df_metricas["Categoria"], df_metricas["F1-Score"], color=colors)
    ax.set_xlabel("F1-Score")
    ax.axvline(x=0.75, color='green', linestyle='--', alpha=0.5, label='Bom (>0.75)')
    ax.axvline(x=0.50, color='orange', linestyle='--', alpha=0.5, label='Regular (>0.50)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Tecnologias utilizadas")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Python 3** — linguagem principal
        - **FastAPI** — framework da API
        - **Scikit-learn** — modelo de ML e TF-IDF
        - **NLTK** — pre-processamento de texto
        """)
    with col2:
        st.markdown("""
        - **Pandas** — manipulação de dados
        - **Streamlit** — interface de apresentação
        - **Uvicorn** — servidor ASGI
        - **Docker** — containerização
        """)

# ==================== CLASSIFICADOR ====================
elif pagina == "Classificador":
    st.title("Classificador de Noticias")
    st.markdown("Digite o título de uma notícia e descubra a categoria em tempo real!")
    st.markdown("---")

    try:
        health = requests.get(f"{API_URL}/health")
        if health.status_code == 200:
            st.success("API conectada e rodando!")
        else:
            st.error("API fora do ar!")
    except:
        st.error("API nao encontrada! Rode o servidor com: uvicorn api.main:app --reload")

    titulo = st.text_input(
        "Título da notícia",
        placeholder="Ex: Lula assina novo decreto sobre economia brasileira"
    )

    if st.button("Classificar", type="primary"):
        if not titulo:
            st.warning("Digite um título para classificar!")
        elif len(titulo.split()) < 3:
            st.warning("O título deve ter pelo menos 3 palavras!")
        else:
            with st.spinner("Classificando..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"title": titulo}
                    )
                    resultado = response.json()
                    categoria = resultado["category"]
                    confianca = resultado["confidence"]

                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"### Categoria: `{categoria}`")
                    with col2:
                        st.info(f"### Confiança: `{confianca:.1%}`")

                    st.markdown("### Top 5 categorias mais prováveis")
                    model_local, vectorizer_local = load_model()
                    titulo_clean = preprocess_text(titulo)
                    titulo_vec = vectorizer_local.transform([titulo_clean])
                    probabilidades = model_local.predict_proba(titulo_vec)[0]
                    categorias = model_local.classes_
                    top5_idx = probabilidades.argsort()[-5:][::-1]
                    top5 = [(categorias[i], probabilidades[i]) for i in top5_idx]

                    for cat, prob in top5:
                        st.progress(float(prob), text=f"{cat}: {prob:.1%}")

                except Exception as e:
                    st.error(f"Erro ao classificar: {e}")
