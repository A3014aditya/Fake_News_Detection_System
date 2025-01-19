import streamlit as st
import joblib
import os
import sys 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model 
import warnings
warnings.filterwarnings('ignore')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 

## Title 
st.title('Fake News Detection System :100:')
News = st.text_area('Enter Your News')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

lemmetizer = WordNetLemmatizer() 

model_lstm = os.path.join('models','lstm.h5')
tokenizer = os.path.join('models','tokenizer.pkl')

## Load the Model LSTM 
model = load_model(model_lstm) 
## Load the Tokenizer Model
tokenize = joblib.load(tokenizer)


pattern = r'\b\d+\b|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|https?://\S+|www\.\S+|[,"\'()-/''?:]'

## Function to preprocess data
def preprocess_data(txt):
    cleantxt = re.sub(pattern,'',txt)
    cleantxt = cleantxt.lower()
    words = word_tokenize(cleantxt)
    words = [lemmetizer.lemmatize(word,pos='v') for word in words if not word in stopwords.words('english')]
    cleantxt = ' '.join(words)
    return cleantxt

if News:
    clean_txt = preprocess_data(News)
    sequences = tokenize.texts_to_sequences([clean_txt])
    padded_sequences = pad_sequences(sequences, maxlen=500,padding='post')


if st.button('Predict'):
    prediction = model.predict(padded_sequences)
    
    if prediction[0][0] <= 0.5:
        st.success('News is Real')
    else:
        st.error('News is Fake') 







