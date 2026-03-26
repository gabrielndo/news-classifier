# Case técnico AeC
Teste técnico - Classificador automático de notícias AeC Centro de Contatos

---

## Etapas de Desenvolvimento

### 1. Análise Exploratória (EDA)

Diante da proposta apresentada, em criar um classificador automático de notícias, foi realizada a Análise exploratória do dataset para entender e filtrar o que iria agregar ao modelo. Essa etapa guiou todas as decisões seguintes.

**Insights encontrados:**

- **167.053 notícias** do jornal Folha UOL, sem nenhuma duplicata
- **6 colunas disponíveis:** `title`, `text`, `date`, `category`, `subcategory`, `link`
- A coluna `subcategory` tinha **137.418 valores nulos** (82% do dataset) — descartada
- A coluna `text` tinha **765 valores nulos** — removidos
- Os títulos têm em média **10 palavras** com distribuição bem comportada (entre 9 e 13 palavras na maioria)
- O dataset apresentava **desbalanceamento severo** entre categorias — `poder` tinha 22.022 notícias enquanto categorias como `2016` tinham apenas 1
- A wordcloud revelou forte presença de termos políticos (Lula, Dilma, Temer, Petrobras, Lava Jato) — coerente com o período de forte conturbação política na coleta dos dados.

**Decisões tomadas a partir do EDA:**
- Usar apenas `title` como entrada — coluna limpa, sem nulos e com tamanho consistente
- Usar `category` como alvo — sem nulos e bem definida
- Filtrar categorias com menos de 1.000 notícias — garantir amostras suficientes para o modelo aprender

---

### 2. Pré-processamento

Os títulos passaram por uma limpeza e padronização antes de chegar ao modelo para garantir consistência de aprendizagem. 

**Etapas aplicadas:**
- Conversão para **minúsculo** — "Brasil" e "brasil" viram a mesma palavra
- Remoção de **pontuação e números** — não agregam valor para classificação
- Remoção de **stopwords em português** — palavras sem significado como "de", "para", "com", "que"

**Exemplo:**
```
Antes:  "Lula diz que vai assinar novo decreto sobre economia"
Depois: "lula diz assinar novo decreto economia"
```

---

### 3. Treinamento do Modelo


**Algoritmo — Regressão Logística:**

Para o treinamento do modelo foi utilizada regressão Logística com vetorização TF-IDF

**Resultados:**

| Métrica | Valor |
|---------|-------|
| Algoritmo | Regressão Logística |
| Vetorização | TF-IDF (10.000 features) |
| Acurácia geral | 74% |
| Notícias de treino | 128.652 |
| Notícias de teste | 32.163 |
| Categorias | 18 |

**Destaques por categoria:**
- `esporte` e `paineldoleitor` → f1-score acima de 0.90 — linguagem muito distinta
- `poder`, `mundo`, `mercado`, `ilustrada` → f1-score entre 0.76 e 0.83 — bom desempenho
- `tv`, `opiniao`, `sobretudo` → f1-score abaixo de 0.30 — categorias com poucas amostras e linguagem parecida com outras

---

### 4. API com FastAPI

O modelo foi embutido em uma API REST com FastAPI.

**Funcionalidades:**
- Validação automática dos dados de entrada
- Retorno da categoria prevista e nível de confiança
- Documentação interativa automática em `/docs`
- Endpoint de saúde para monitoramento

---

 Bibliotecas:
| `scikit-learn` | TF-IDF e Regressão Logística |
| `nltk` | Stopwords com suporte em português|
| `matplotlib` / `seaborn` | Visualizações no EDA |
| `wordcloud` | Nuvem de palavras |
| `fastapi` | API sugerida para familiarização |
| `uvicorn` | Servidor ASGI necessário para rodar o FastAPI |
| `joblib` | Salvar e carregar o modelo treinado em arquivo `.pkl` |
| `jupyter` | Notebook interativo para o EDA com gráficos online |

---

## Estrutura do Projeto
```
news-classifier/
├── data/               # Dataset de notícias (não versionado)
├── notebooks/
│   └── eda.ipynb       # Análise exploratória completa
├── src/
│   └── train.py        # Treinamento do modelo
├── api/
│   └── main.py         # API FastAPI
├── models/             # Modelo e vectorizer salvos (não versionados)
├── requirements.txt    # Dependências do projeto
├── Dockerfile          # Configuração do container
└── README.md
```

---

## Como acessar a API

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

### Passo 4 — Rode a API
```bash
uvicorn api.main:app --reload
```

### Passo 5 — Acesse a API
Abra no navegador:
```
http://localhost:8000/docs
```
A documentação interativa já estará disponível para testar os endpoints! 

---

## Como rodar com Docker

### Pré-requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando

### Passo 1 — Clone o repositório
```bash
git clone https://github.com/gabrielndo/news-classifier.git
cd news-classifier
```

### Passo 2 — Build da imagem
```bash
docker build -t news-classifier .
```

### Passo 3 — Rode o container
```bash
docker run -p 8000:8000 news-classifier
```

### Passo 4 — Acesse a API
Abra no navegador:
```
http://localhost:8000/docs
```
A documentação interativa já estará disponível para testar os endpoints.


## Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Verifica se a API está no ar |
| GET | `/health` | Verifica a saúde da API |
| POST | `/predict` | Classifica uma notícia |

---

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

---

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

---

## Tecnologias utilizadas

- **Python 3** — linguagem principal
- **FastAPI** — framework da API
- **Scikit-learn** — modelo de ML e TF-IDF
- **NLTK** — pré-processamento de texto em português
- **Pandas** — manipulação de dados
- **Uvicorn** — servidor ASGI
- **Docker** — containerização