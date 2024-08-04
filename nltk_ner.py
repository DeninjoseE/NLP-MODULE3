import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import requests
import matplotlib.pyplot as plt
from collections import Counter

# Download necessary NLTK resources
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Define your NewsAPI key
NEWS_API_KEY = '9eb6cc7a95f6415ea8c2e1646a3b12ca'

# Fetch a news article
def fetch_news_article(api_key, query):
    url = f'https://newsapi.org/v2/top-headlines?q={query}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if articles:
            title = articles[0].get('title', '')
            description = articles[0].get('description', '')
            return f"{title}. {description}".strip()
    return None

# Extract entities using NLTK (rule-based approach)
def extract_entities_nltk(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    chunked = ne_chunk(tagged)
    entities = []
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            entities.append((entity, entity_type))
    return entities

# Visualize NLTK entities
def visualize_nltk_entities(entities):
    entity_counts = Counter([entity[1] for entity in entities])
    labels, values = zip(*entity_counts.items())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='skyblue')
    plt.title('Frequency of Entity Types')
    plt.xlabel('Entity Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()

# Main function
def main():
    # Input a unique query to fetch a specific news article
    query = input("Enter a unique keyword to fetch the article (e.g., your name): ")

    # Fetch a news article
    article = fetch_news_article(NEWS_API_KEY, query)
    if not article:
        print("No article found with the given query.")
        return

    print("Article Text:")
    print(article)

    # Extract entities using NLTK
    nltk_entities = extract_entities_nltk(article)
    print("\nEntities extracted using NLTK:")
    for entity in nltk_entities:
        print(f"{entity[0]} -> {entity[1]}")

    # Visualize entities using matplotlib
    visualize_nltk_entities(nltk_entities)

if __name__ == "__main__":
    main()
