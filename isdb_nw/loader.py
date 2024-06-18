import os
import wget
import zipfile
import sys
import spacy
import torch
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import gensim, logging
from navec import Navec
from transformers import AutoTokenizer, AutoModel


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def load_w2v():
    if not os.path.exists('data/220.zip'):
        print('\nGonna load wav2vec...')
        model_url = 'http://vectors.nlpl.eu/repository/20/220.zip'
        m = wget.download(model_url, bar=bar_progress, out='data')
        model_file = 'data/' + model_url.split('/')[-1]
        with zipfile.ZipFile(model_file, 'r') as archive:
            stream = archive.extract('model.bin', path='data')
            model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format('data/model.bin', binary=True)
        return model


def load_navec():
    if not os.path.exists('data'):
       os.mkdir('data')
    model_url = 'https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar'
    path = 'data/' + model_url.split('/')[-1]
    if not os.path.exists(path):
        print('Gonna load navec...')
        m = wget.download(model_url, bar=bar_progress, out=path)

    model = Navec.load(path)
    return model


def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("/tmp/model")
    model = AutoModel.from_pretrained("/tmp/model")
    return tokenizer, model


def load_spacy():
    return spacy.load("ru_core_news_sm")


def main():
    load_w2v()
    load_navec()
    load_spacy()
    load_bert()


if __name__ == '__main__':
    main()