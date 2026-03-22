import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess_text(text):
    # Converte para minúsculo
    text = text.lower()
    
    # Remove caracteres especiais e números
    text = re.sub(r'[^a-záéíóúâêîôûãõçàü\s]', '', text)
    
    # Remove espaços extras
    text = text.strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('portuguese'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

def load_data(path='data/articles.csv'):
    # Carrega o dataset
    df = pd.read_csv(path)
    
    # Remove linhas com título ou categoria nulos
    df = df.dropna(subset=['title', 'category'])
    
    # Filtra apenas categorias com mais de 1000 notícias
    categorias_validas = df['category'].value_counts()
    categorias_validas = categorias_validas[categorias_validas >= 1000].index
    df = df[df['category'].isin(categorias_validas)]
    
    # Aplica o pré-processamento nos títulos
    df['title_clean'] = df['title'].apply(preprocess_text)
    
    return df

def train_model(df):
    # Separa features e target
    X = df['title_clean']
    y = df['category']
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Cria e aplica o TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Treina o modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Avalia o modelo
    y_pred = model.predict(X_test_vec)
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer

def save_model(model, vectorizer):
    # Salva o modelo treinado
    joblib.dump(model, 'models/model.pkl')
    
    # Salva o vectorizer
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    
    print("Modelo e vectorizer salvos em models/")

if __name__ == '__main__':
    print("Carregando e processando dados...")
    df = load_data()
    
    print(f"Dataset filtrado: {df.shape[0]} notícias")
    print(f"Categorias: {df['category'].unique()}")
    
    print("\nTreinando modelo...")
    model, vectorizer = train_model(df)
    
    print("\nSalvando modelo...")
    save_model(model, vectorizer)
    
    print("\nConcluído!")