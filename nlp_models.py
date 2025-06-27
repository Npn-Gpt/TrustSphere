from transformers import pipeline, DistilBertTokenizer, DistilBertModel

# Sentiment model
sentiment_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Embedding model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
embedding_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
