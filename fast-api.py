# Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.corpus import wordnet 

# Ensure NLTK resources are downloaded
nltk.download('punkt')  # Tokenization
nltk.download('stopwords')  # Stop words
nltk.download('averaged_perceptron_tagger_eng')  # POS tagging
nltk.download('wordnet')  # WordNet for lemmatization
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Load the sentiment analysis model and tokenizer
model = load_model('saved_models/sentiment_analysis_model.h5')

with open('saved_models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for the input data
class TextInput(BaseModel):
    text: str

# Preprocessing functions
def remove_usernames(text):
    # Remove usernames (mentions) from the text
    return re.sub(r'@\w+', '', text)

def tokenize_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    return tokens

def normalize_tokens(tokens):
    # Normalize tokens to lowercase
    normalized_tokens = [token.lower() for token in tokens]
    return normalized_tokens

def clean_tokens(tokens):
    # Filter out non-alphanumeric tokens
    cleaned_tokens = [token for token in tokens if token.isalnum() and not token.isdigit()]
    return cleaned_tokens

def remove_stopwords(tokens):
    # Remove stop words from tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def lemmatize_tokens(tokens):
    # Lemmatize tokens based on part of speech
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens:
        pos_tag = nltk.pos_tag([token])[0][1][0].upper()
        pos_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}
        wordnet_pos = pos_map.get(pos_tag, wordnet.NOUN)
        lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
        lemmatized_tokens.append(lemma)
    return lemmatized_tokens

def filter_meaningful_tokens(tokens):
    # Filter tokens based on length and presence in WordNet
    meaningful_tokens = [token for token in tokens if len(token) >= 3]
    return meaningful_tokens

def preprocess_text(text):
    # Step 1: Remove usernames
    text = remove_usernames(text)

    # Step 2: Tokenize text
    tokens = tokenize_text(text)
    
    # Step 3: Normalize tokens
    normalized_tokens = normalize_tokens(tokens)
    
    # Step 4: Clean tokens
    cleaned_tokens = clean_tokens(normalized_tokens)
    
    # Step 5: Remove stop words
    filtered_tokens = remove_stopwords(cleaned_tokens)
    
    # Step 6: Lemmatize tokens
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    
    # Step 7: Filter meaningful tokens
    meaningful_tokens = filter_meaningful_tokens(lemmatized_tokens)
    
    # Join tokens into preprocessed text
    preprocessed_text = ' '.join(meaningful_tokens)
    
    return preprocessed_text

# Define a prediction endpoint
@app.post('/predict/')
async def predict_sentiment(input_data: TextInput):
    # Preprocess the input text
    cleaned_text = preprocess_text(input_data.text)
    
    # Convert the cleaned text to sequences
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=30)  # Pad sequences to ensure uniform length
    
    # Predict sentiment
    prediction = model.predict(padded_sequences)
    confidence_score = prediction[0][0]  # Get the confidence score for the positive class
    sentiment = 'Positive' if confidence_score > 0.5 else 'Negative'  # Binary classification based on threshold
    
    return {
        'text': input_data.text,
        'sentiment': sentiment,
        'confidence': float(confidence_score)  # Return confidence score as a float
    }
