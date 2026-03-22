# News Classifier

Você já se perguntou como portais de notícias conseguem organizar milhares de artigos por categoria automaticamente? Esse projeto faz exatamente isso!

O **News Classifier** é um classificador automático de notícias brasileiras. Basta enviar o título de uma notícia e ele responde qual categoria ela pertence — política, esporte, economia, tecnologia e mais 14 categorias.

## Como funciona?

O projeto passa por 4 etapas principais:

### 1. Análise Exploratória (EDA)
Antes de qualquer coisa, exploramos o dataset para entender com o que estávamos trabalhando:
- 167.053 notícias do jornal Folha UOL
- 18 categorias diferentes
- Títulos com média de 10 palavras
- Dataset limpo, sem duplicatas

Essa etapa foi fundamental para tomar decisões como quais categorias incluir no modelo e qual coluna usar como entrada.

### 2. Pré-processamento
Os títulos passam por uma limpeza antes de chegar ao modelo:
- Tudo convertido para minúsculo
- Remoção de pontuação e números
- Remoção de stopwords em português ("de", "para", "com"...)

Isso é necessário porque o modelo não entende linguagem — ele precisa de texto padronizado e limpo.

### 3. Treinamento
- Transformamos os títulos em números usando **TF-IDF**
- Treinamos uma **Regressão Logística** com 128.652 notícias
- Resultado: **74% de acurácia** em 18 categorias

### 4. API
Embrulhamos tudo em uma API com FastAPI. Qualquer sistema pode mandar um título e receber a categoria de volta em milissegundos.

## Estrutura do Projeto
```
news-classifier/
├── data/               # Dataset de notícias
├── notebooks/
│   └── eda.ipynb       # Análise exploratória
├── src/
│   └── train.py        # Treinamento do modelo
├── api/
│   └── main.py         # API FastAPI
├── models/             # Modelo e vectorizer salvos
├── requirements.txt
├── Dockerfile
└── README.md
```

## ⚙️ Como rodar o projeto

### Pré-requisitos
- Python 3.8+
- Git

### Passo 1 — Clone o repositório
```bash
git clone https://github.com/gabrielndo/news-classifier.git
cd news-classifier
```

### Passo 2 — Crie e ative o ambiente virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Passo 3 — Instale as dependências
```bash
pip install -r requirements.txt
```

### Passo 4 — Baixe o dataset
Acesse o [Kaggle](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol), baixe o dataset e coloque o arquivo `articles.csv` dentro da pasta `data/`.

### Passo 5 — Treine o modelo
```bash
python src/train.py
```
Esse passo pode demorar alguns minutos. Ao final, os arquivos `model.pkl` e `vectorizer.pkl` serão salvos na pasta `models/`.

### Passo 6 — Rode a API
```bash
uvicorn api.main:app --reload
```

Acesse a documentação interativa em: http://127.0.0.1:8000/docs

### Passo 7 (opcional) — Rode com Docker
```bash
docker build -t news-classifier .
docker run -p 8000:8000 news-classifier
```

## 📡 Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Verifica se a API está no ar |
| GET | `/health` | Verifica a saúde da API |
| POST | `/predict` | Classifica uma notícia |

## Exemplo de uso

**Requisição:**
```json
{
  "title": "Lula sanciona nova lei sobre economia"
}
```

**Resposta:**
```json
{
  "title": "Lula sanciona nova lei sobre economia",
  "category": "poder",
  "confidence": 0.5173
}
```

## Categorias disponíveis

| Categoria | Descrição |
|-----------|-----------|
| poder | Política |
| mercado | Economia |
| esporte | Esportes |
| mundo | Internacional |
| cotidiano | Dia a dia |
| ilustrada | Cultura |
| colunas | Colunas de opinião |
| tec | Tecnologia |
| educacao | Educação |
| turismo | Turismo |
| ciencia | Ciência |
| tv | Televisão |
| opiniao | Opinião |
| saopaulo | São Paulo |
| equilibrioesaude | Saúde |
| paineldoleitor | Cartas dos leitores |
| ilustrissima | Cultura aprofundada |
| sobretudo | Coluna específica |

## Sobre o modelo

A escolha da **Regressão Logística** com **TF-IDF** foi intencional — é uma combinação simples, rápida e muito eficiente para classificação de texto. O foco do projeto foi entregar algo funcional e bem estruturado do início ao fim, não necessariamente o modelo com maior acurácia.

| Métrica | Valor |
|---------|-------|
| Algoritmo | Regressão Logística |
| Vetorização | TF-IDF (10.000 features) |
| Acurácia geral | 74% |
| Notícias de treino | 128.652 |
| Categorias | 18 |

## 🛠️ Tecnologias utilizadas

- **Python 3** — linguagem principal
- **FastAPI** — framework da API
- **Scikit-learn** — modelo de ML e TF-IDF
- **NLTK** — pré-processamento de texto
- **Pandas** — manipulação de dados
- **Uvicorn** — servidor ASGI
- **Docker** — containerização