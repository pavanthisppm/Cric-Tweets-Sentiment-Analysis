 
import pytorch_pretrained_bert as ppb
assert 'bert-large-cased' in ppb.modeling.PRETRAINED_MODEL_ARCHIVE_MAP
 
import pandas as pd
import streamlit as st
import json
import requests  

from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from streamlit_lottie import st_lottie 
 


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_alhjvesz.json")

st_lottie(
    lottie,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",   
    height='75px',
    width='75px',
    key=None,
)

st.header('Cric-Tweets Sentiment Analysis')

sentiment_model = pipeline(model="sppm/cric-tweets-sentiment-analysis")
with st.expander('Analyze Tweet'):
    text = st.text_input('Tweet here: ')
    if text:
        senti = list(sentiment_model(text)[0].values())[0]
        if senti=='LABEL_1':
            label = senti.replace(senti, 'Positive')
        elif senti=='LABEL_0':
            label = senti.replace(senti, 'Negative')
        st.write('Label: ', label)
       
 
with st.expander('Analyze CSV'):
    
    upl = st.file_uploader('Upload file: ')
 
    if upl:
        df = pd.read_csv(upl, encoding='latin1')
        del df['Unnamed: 0']
        b=[]
        for i in range(len(df)):
            b.append(list(sentiment_model(df['Tweet'][i])[0].values())[0])
        df['Sentiment'] = b
        # Convert label1/label2 string columns into int columns of 1/0
        df[['Sentiment']] = \
        (df[['Sentiment']] == 'LABEL_1').astype(int)
        st.write(df.head())
        st.write('Number of Positive Tweets: ', df.Sentiment.sum())
        st.write('Total Number of Tweets: ',  len(df))

        @st.cache
        def convert_df(df):
           
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name= upl.name,
            mime='text/csv',
        )
