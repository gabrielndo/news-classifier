import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from wordcloud import WordCloud
from nltk.corpus import stopwords

nltk.download('stopwords')

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Configuração da página
st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="wide"
)

# Carrega o modelo e vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return model, vectorizer

# Carrega o dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/articles.csv')
    return df

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

# Sidebar com navegação
st.sidebar.image("https://img.icons8.com/emoji/96/newspaper-emoji.png", width=80)
st.sidebar.title("News Classifier")
st.sidebar.markdown("Classificador automático de notícias")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegação",
    ["Sobre mim", "Início", "Análise dos Dados", "Sobre o Modelo","Classificador"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por **Gabriel**")
st.sidebar.markdown("Teste Técnico — AeC Centro de Contatos")
if pagina == "Sobre mim":
    # Cabeçalho
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("assets/foto.png", width=250)
    
    with col2:
        st.title("Gabriel Nantes de Oliveira")
        st.markdown("#### Cientista de Dados Jr.")
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

    # Stacks
    st.markdown("### Stacks")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div style='text-align:center; padding:15px; background:#1e1e2e; border-radius:10px'>
            <h4>Python</h4>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:15px; background:#1e1e2e; border-radius:10px'>
            <h4>SQL</h4>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='text-align:center; padding:15px; background:#1e1e2e; border-radius:10px'>
            <h4>Machine Learning</h4>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style='text-align:center; padding:15px; background:#1e1e2e; border-radius:10px'>
            <h4>Power BI</h4>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div style='text-align:center; padding:15px; background:#1e1e2e; border-radius:10px'>
            <h4>Data Analysis</h4>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Projetos
    st.markdown("### Projetos")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='padding:20px; background:#1e1e2e; border-radius:10px; height:180px'>
            <h4>Previsao de Falhas</h4>
            <p>Deteccao de falhas em equipamentos por analise de vibração usando Machine Learning.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='padding:20px; background:#1e1e2e; border-radius:10px; height:180px'>
            <h4>Agente de Pendencias</h4>
            <p>Agente inteligente para extracao e categorizacao automatica de falhas e pendencias.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='padding:20px; background:#1e1e2e; border-radius:10px; height:180px'>
            <h4>Monitor de Suplementos</h4>
            <p>Aplicativo para lembrete e monitoramento de suplementacao diaria.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Mapa de habilidades (radar)
    st.markdown("### Mapa de Soft Skills")

    import numpy as np
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.pyplot as plt

    skills = ['Inteligencia\nEmocional', 'Criatividade', 'Raciocinio\nLogico', 
              'Empatia', 'Adaptabilidade']
    valores = [90, 85, 92, 88, 87]

    # Fecha o radar
    valores += valores[:1]
    N = len(skills)
    angulos = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(polar=True))
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')

    ax.plot(angulos, valores, 'o-', linewidth=2, color='#4fa3e0')
    ax.fill(angulos, valores, alpha=0.25, color='#4fa3e0')
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(skills, color='white', size=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.grid(color='grey', alpha=0.3)
    ax.spines['polar'].set_color('grey')

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.pyplot(fig)

# Página Início
if pagina == "Início":
    st.title("News Classifier")
    st.markdown("### Classificador automático de notícias brasileiras")
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
        - **Confiança** — o nível de certeza do modelo na predição
        
        Tudo isso em **milissegundos**!
        """)

    st.markdown("---")
    st.info("Use o menu lateral para navegar entre as seções")

# Página Análise dos Dados
elif pagina == "Análise dos Dados":
    st.title("Análise dos Dados")
    st.markdown("Insights extraídos durante a análise exploratória do dataset.")
    st.markdown("---")

    try:
        df = load_data()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de notícias", f"{len(df):,}")
        with col2:
            st.metric("Categorias únicas", df['category'].nunique())
        with col3:
            st.metric("Período", f"{df['date'].min()[:4]} - {df['date'].max()[:4]}")

        st.markdown("---")
        st.markdown("### Distribuição de Categorias")
        fig, ax = plt.subplots(figsize=(12, 4))
        df['category'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel("Categoria")
        ax.set_ylabel("Quantidade")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Palavras mais frequentes nos títulos")
        stop_words = set(stopwords.words('portuguese'))
        texto = ' '.join(df['title'].dropna().values)
        wc = WordCloud(
            width=800,
            height=400,
            stopwords=stop_words,
            background_color='white'
        ).generate(texto)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Distribuição do tamanho dos títulos")
        df['title_len'] = df['title'].dropna().apply(lambda x: len(x.split()))
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['title_len'], bins=30, ax=ax, color='steelblue')
        ax.set_xlabel("Número de palavras")
        ax.set_ylabel("Frequência")
        plt.tight_layout()
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Media de palavras", f"{df['title_len'].mean():.0f}")
        with col2:
            st.metric("Minimo", f"{df['title_len'].min():.0f}")
        with col3:
            st.metric("Maximo", f"{df['title_len'].max():.0f}")

    except:
        st.warning("Dataset não encontrado! Coloque o arquivo `articles.csv` na pasta `data/` para visualizar a análise.")

# Página Sobre o Modelo
elif pagina == "Sobre o Modelo":
    st.title("Sobre o Modelo")
    st.markdown("---")

    st.markdown("### Metodologia")
    st.markdown("""
    A escolha da **Regressão Logística** com **TF-IDF** foi intencional — é uma combinação 
    simples, rápida e muito eficiente para classificação de texto. O foco do projeto foi 
    entregar algo funcional e bem estruturado do início ao fim.
    """)

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
        
elif pagina == "Classificador":
    st.title("Classificador de Noticias")
    st.markdown("Digite o título de uma notícia e descubra a categoria!")
    st.markdown("---")

    # Verifica se a API está no ar
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

                    # Top 5 categorias
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
