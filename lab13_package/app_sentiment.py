import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go

nltk.download('stopwords')
nltk.download('wordnet')

def visualize_probability_plotly(probability, sentiment, label="Probability"):
    label = label+" of "+sentiment
    color = "green"
    if sentiment == "negative":
        color = "red"
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=probability,
        title={'text': label},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': color}}
    ))
    st.plotly_chart(fig)


stopwords_set = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Lowercase text
    text = text.lower()
    # Tokenize text
    tokens = text.split()
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords_set]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a string
    return ' '.join(tokens)

# Load the saved model, vectorizer, and label encoder
try:
    model, vectorizer, label_encoder = joblib.load('sentiment_model_bow.joblib')
except FileNotFoundError:
    model, vectorizer, label_encoder = joblib.load('sentiment_model_tfidf.joblib')

st.title('Sentiment Analysis App')

user_input = st.text_area('Enter your review here:')

if st.button('Predict'):
    if user_input:
        preprocessed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([preprocessed_input])
        prediction = model.predict(vectorized_input)[0]
        sentiment = label_encoder.inverse_transform([prediction])[0]
        probs = model.predict_proba(vectorized_input)
        st.write(f'Predicted Sentiment: {sentiment} with probability {(probs[0][prediction])}')
        visualize_probability_plotly(probs[0][prediction], sentiment)
    else:
        st.write('Please enter a review.')