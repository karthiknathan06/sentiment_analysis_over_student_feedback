from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import nltk
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())
from nltk.tokenize import RegexpTokenizer
import re
import matplotlib.pyplot as plt

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

 
@app.route('/')
def home():
    return render_template("dash.html")

@app.route('/feed')
def feed():
    return render_template("page.html")

@app.route("/analyse")
def analyse():
    column=['feedback']
    df=pd.read_csv("Feedback.csv",names=column,header=None)
    fa=df['feedback']
    df['feedback']=df['feedback'].astype(str)
    df.dropna(subset=['feedback'])
    dict={'positive':0,'negative':0,'neutral':0}
    positive=[]
    negative=[]
    neutral=[]

    for i in fa:
        #remove punctuation
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(i)
        final=' '.join(tokens)
        
        #remove non english words
        a1=" ".join(w for w in nltk.wordpunct_tokenize(final)if w.lower() in words or not w.isalpha())
        
        #remove proper nouns
        tokenized = nltk.word_tokenize(a1)
        pos=nltk.tag.pos_tag(tokenized)
        ed=[word for word,tag in pos if tag!='NNP' and tag!='NNPs']#removing proper nouns
        end=' '.join(ed)
        
        #remove stop words
        en=[i for i in word_tokenize(end.lower()) if i not in stop] 
        final=' '.join(en)
        
        #using vader classfiers
        sid=SentimentIntensityAnalyzer()
        ss=sid.polarity_scores(final)
        if(ss['compound']==0):
            dict['neutral']+=1
            neutral.append(final)
        elif(ss['compound']>0):
            dict['positive']+=1
            positive.append(final)
        else:
            dict['negative']+=1
            negative.append(final)
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [dict['positive'],dict['negative'],dict['neutral']]
    print(dict['positive'])
    print(dict['negative'])
    print(dict['neutral'])
    values=[ k for k in dict.values()]
    labels=["Positive","Neutral","negative"]
    colors=["#F7464A","#46BFBD","#FDB45C"]
    return render_template( 'analyse.html',set=zip(values,colors))
 
if __name__ == "__main__":
    app.run()