import streamlit as st
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return "Positive - All Good!"
    elif polarity < 0:
        return "Negative - All Bad!"
    else:
        return "Neutral - It's Okay!"

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter text to analyze its sentiment.")

user_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    if user_input.strip():
        sentiment = get_sentiment(user_input)
        st.write(f"### Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")
