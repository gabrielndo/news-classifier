from fastapi import FastAPI
from pydantic import BaseModel
import  joblib
import  re 
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Inicializa a aplicação
app = FastAPI(
    title="News Classifier API",
    description="API para classificação de categorias de notícias brasileiras",
    version="1.0.0"
)

# Carrega o modelo e vectorizer treinados
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Carrega stopwords
stop_words = set(stopwords.words('portuguese'))

# Define o formato dos dados de entrada
class NewsInput(BaseModel):
    title: str

# Função de pré-processamento
def preprocess_text(text):
    # Converte para minúsculo
    text = text.lower()
    
    # Remove caracteres especiais e números
    text = re.sub(r'[^a-záéíóúâêîôûãõçàü\s]', '', text)
    
    # Remove espaços extras
    text = text.strip()
    
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

# Endpoint de verificação
@app.get("/")
def root():
    return {"message": "News Classifier API está rodando!"}

# Endpoint de saúde
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint de classificação
@app.post("/predict")
def predict(news: NewsInput):
    # Pré-processa o título
    title_clean = preprocess_text(news.title)
    
    # Transforma em vetor
    title_vec = vectorizer.transform([title_clean])
    
    # Faz a predição
    category = model.predict(title_vec)[0]
    
    # Pega a probabilidade de cada categoria
    probabilities = model.predict_proba(title_vec)[0]
    confidence = round(float(max(probabilities)), 4)
    
    return {
        "title": news.title,
        "category": category,
        "confidence": confidence
    }