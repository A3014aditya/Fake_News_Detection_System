import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model 

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

lemmetizer = WordNetLemmatizer() 

## Load the Model LSTM 
model = load_model('models\lstm.h5')
## Load the Tokenizer Model
tokenize = joblib.load(R'D:\Fake_News_Detection(new)\models\tokenizer.pkl')

## Title 
st.title('Fake News Detection System :100:')

News = st.text_area('Enter Your News')

## Function to preprocess data
def preprocess_data(txt):
    cleantxt = re.sub(r'http\S+|www\.\S+', '',txt)  # remove links
    cleantxt = re.sub(r'@\S+','',cleantxt)          # remove mentions
    cleantxt = re.sub(r'#\S+','',cleantxt)          # remove hashtags
    cleantxt = re.sub(r'[^\w\s]','',cleantxt)       # remove white spaces
    cleantxt = re.sub('[^a-zA-Z]',' ',cleantxt)
    cleantxt = cleantxt.lower()
    words = word_tokenize(cleantxt)
    words = [lemmetizer.lemmatize(word,pos='v') for word in words if not word in stopwords.words('english')]
    cleantxt = ' '.join(words)
    return cleantxt


if st.button('Predict'):
    cleaned_txt = preprocess_data(News)
    
    sequences = tokenize.texts_to_sequences([cleaned_txt])
    padded_sequences = pad_sequences(sequences, maxlen=500,padding='post')
    
    prediction = model.predict(padded_sequences)
    
    if prediction[0][0] <= 0.5:
        st.success('News is Real')
    else:
        st.error('News is Fake') 







