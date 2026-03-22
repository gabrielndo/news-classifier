# Imagem base do Python
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Baixa as stopwords do NLTK
RUN python -c "import nltk; nltk.download('stopwords')"

# Expõe a porta 8000
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]