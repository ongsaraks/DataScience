import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load the saved model and vectorizer
try:
    model, vectorizer = joblib.load('disaster_detection_model.joblib')
except FileNotFoundError:
    st.error("Error: 'disaster_detection_model.joblib' not found. Make sure you have run Task 02 and saved the model.")
    st.stop()

st.title('Natural Disaster Tweet Detector')

user_input = st.text_area("Enter a tweet to check for disaster relevance:")

if st.button('Predict'):
    if user_input:
        processed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.warning("This tweet is predicted to be about a natural disaster.")
        else:
            st.success("This tweet is predicted to NOT be about a natural disaster.")

        # Optional: Display probability if the model has predict_proba
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vectorized_input)[0]
            prob_disaster = probabilities[1] * 100
            prob_not_disaster = probabilities[0] * 100
            st.write(f"Probability of being a disaster tweet: {prob_disaster:.2f}%")
            st.write(f"Probability of not being a disaster tweet: {prob_not_disaster:.2f}%")
    else:
        st.warning("Please enter a tweet.")