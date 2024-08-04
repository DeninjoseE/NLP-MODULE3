import spacy
import requests
from spacy import displacy

# Load SpaCy's pre-trained model
nlp = spacy.load('en_core_web_sm')

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

# Extract entities using SpaCy
def extract_entities_spacy(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return doc, entities

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

    # Extract entities using SpaCy
    doc, spacy_entities = extract_entities_spacy(article)
    print("\nEntities extracted using SpaCy:")
    for entity in spacy_entities:
        print(f"{entity[0]} -> {entity[1]}")

    # Visualize entities using SpaCy's displacy with custom host and port
    displacy.serve(doc, style="ent", host="127.0.0.1", port=8080)

if __name__ == "__main__":
    main()
