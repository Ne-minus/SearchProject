import pandas as pd
import numpy as np
import spacy
import zipfile
import wget
import sys
import gensim, logging
import re
import os
import sys
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors


def preprocess_text(text, nlp):
    lemmas_only = ''
    lemmas_and_pos = ''
    text = re.sub(r'\s*[A-Za-z]+\b', '', text)
    text = nlp(text)
    for token in text:
        if token.is_alpha and not token.is_stop:
            lemmas_only += f'{token.lemma_} '
            lemmas_and_pos += f'{token.lemma_}_{token.pos_} '
    return lemmas_only, lemmas_and_pos


if __name__ == '__main__':
    df = pd.read_csv('/Users/eneminova/pythonProject/pythonProject/lenta-ru-news.csv')
    topics = ['Россия', 'Мир', 'Бизнес', 'Экономика']
    df = df[df['topic'].isin(topics)]
    df = df.sample(frac=1).head(2000)
    nlp = spacy.load("ru_core_news_sm")
    df[['clean', 'clean_pos']] = df.text.apply(lambda x: pd.Series(preprocess_text(x, nlp)))
    df.to_csv('isdb_hw2.csv')