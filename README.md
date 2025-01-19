# Fake News Detection Using Deep Learning

This project focuses on detecting fake news using deep learning models, including Simple RNN, LSTM RNN, GRU RNN, and Bidirectional RNN. The goal is to classify news articles as either real or fake based on their content.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Models](#models)
5. [Dependencies](#dependencies)
6. [Usage](#usage)

## Introduction

The rise of fake news on social media and the internet has led to significant challenges in identifying truthful information. This project employs various deep learning models to detect fake news and help mitigate the spread of misinformation.

## Dataset

The dataset used for this project consists of labeled news articles. Each article is classified as either `Real` or `Fake`. 

### Sources:
- [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

### Structure:
The dataset should contain the following columns:
- `text`: The body of the news article.
- `label`: The classification (`Real` or `Fake`).

## Features

The primary features extracted for this project include:
- Text preprocessing: Tokenization, removing stop words, and lemmatization.
- Vectorization: Representing text using word embeddings.

## Models

This project implements the following deep learning models:

1. **Simple RNN**: A basic recurrent neural network to understand sequential text data.
2. **LSTM RNN**: Long Short-Term Memory networks to capture long-term dependencies in text.
3. **GRU RNN**: Gated Recurrent Unit networks, a simpler alternative to LSTM.
4. **Bidirectional RNN**: Processes text in both forward and backward directions for better context.

Each model is evaluated on the same dataset to compare performance. After comparison LSTM RNN gives a good performance. 

## Dependencies

Install the required libraries using the command below:

```bash
pip install -r requirements.txt
```

### Requirements:
- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit 
- Matplotlib
- NLTK
- Wordcloud 

## Usage

### 1. Clone the Repository:


    git clone https://github.com/A3014aditya/Fake_News_Detection_System.git

### 2. Create a new virtual environment:


    Conda create -p venv  python==3.9 -y

### 3. Activate  virtual environment:


    Conda activate ./venv 

### 4 Install the dependencies:


    pip install -r requirements.txt 

### 5. Run application :


    streamlit run app.py

# User Interface 
- Link https://aditya-fake-news-detection.streamlit.app/ 
![Image](https://github.com/user-attachments/assets/38eb59d7-13df-4b24-b88e-5f807bc7d8e0) 



