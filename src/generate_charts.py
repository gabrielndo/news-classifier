import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import os

nltk.download('stopwords')

# Cria a pasta assets se não existir
os.makedirs('assets', exist_ok=True)

# Carrega o dataset
print("Carregando dataset...")
df = pd.read_csv('data/articles.csv')

# Gráfico 1 — Distribuição de categorias
print("Gerando gráfico de categorias...")
fig, ax = plt.subplots(figsize=(12, 4))
df['category'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel("Categoria")
ax.set_ylabel("Quantidade")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('assets/distribuicao_categorias.png', dpi=150, bbox_inches='tight')
plt.close()
print("Salvo: assets/distribuicao_categorias.png")

# Gráfico 2 — Wordcloud
print("Gerando wordcloud...")
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
plt.tight_layout()
plt.savefig('assets/wordcloud.png', dpi=150, bbox_inches='tight')
plt.close()
print("Salvo: assets/wordcloud.png")

# Gráfico 3 — Distribuição do tamanho dos títulos
print("Gerando histograma...")
df['title_len'] = df['title'].dropna().apply(lambda x: len(x.split()))
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df['title_len'], bins=30, ax=ax, color='steelblue')
ax.set_xlabel("Número de palavras")
ax.set_ylabel("Frequência")
plt.tight_layout()
plt.savefig('assets/distribuicao_titulos.png', dpi=150, bbox_inches='tight')
plt.close()
print("Salvo: assets/distribuicao_titulos.png")

print("\nTodos os gráficos gerados com sucesso!")