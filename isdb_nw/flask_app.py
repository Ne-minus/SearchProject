import os
import re
import pandas as pd
import time
import sqlalchemy
from flask import Flask, render_template, request, redirect, url_for
from enum import Enum
# from flask_sqlalchemy import SQLAlchemy
from db_design import Texts, Session
from index import AllModels
from loader import load_navec, load_spacy, load_w2v, load_bert


class SearchType(Enum):
    """
    Defines search type
    """

    bm25 = 'bm25'
    navec = 'navec'
    w2v = 'word2vec'
    bert = 'bert'


def unpacking(column) -> list:
    """
    Unpack values from database

    Args:
        column: A column extracted from database

    Returns:
        texts : list
            A list of texts
    """

    return [text[0] for text in column]


def download_models():
    """

    Returns:
        Function returns model objects loaded to RAM

    """

    navec_load = load_navec()
    w2v_load = load_w2v()
    tokenizer_load, bert_load = load_bert()
    nlp_load = load_spacy()

    return navec_load, w2v_load, nlp_load, tokenizer_load, bert_load


def perform_search(query: str, metric: str, number: int) -> list:
    """
    Performs search using search method depending on metric provided

    Args:
        query: str
            A query provided by user
        metric: str
            A metric chosen by user
        number: str
            The size of results (the number of texts to show)

    Returns:
        result: list
            List of matching text identifiers
    """

    if metric == SearchType.bm25:
        result = all_models.bm25_i.search(query, n=number)
    elif metric == SearchType.navec:
        result = all_models.navec_i.search(query, n=number)
    elif metric == SearchType.w2v:
        result = all_models.word2v_i.search(query, n=number)
    else:
        result = all_models.bert_i.search(query, n=number)

    return result


TEMPLATE_DIR = os.path.abspath('./templates')
STATIC_DIR = os.path.abspath('./templates/static')
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

session = Session()
raw_texts = unpacking(session.query(Texts.text).order_by(Texts.index))
processed = unpacking(session.query(Texts.clean).order_by(Texts.index))
processed_pos = unpacking(session.query(Texts.clean_pos).order_by(Texts.index))

navec, w2v, nlp, tokenizer, bert = download_models()
all_models = AllModels(processed, processed_pos, raw_texts, navec, w2v, tokenizer, bert, nlp)


@app.route('/')
def main_page():
    return render_template('main_page.html')


@app.route('/search')
def search():
    return render_template('index.html')


@app.route('/data', methods=['get', 'post'])
def data():
    """
    Receives data from user and sends relevant results

    Returns:
        Renders template with matching text and accompanying info
    """

    if not request.form:
        return redirect(url_for('/search'))

    answer = request.form
    query, metric, number = answer['query'], SearchType(answer['metric']), int(answer['size'])

    start = time.time()
    search_res = perform_search(query, metric, number)
    end = time.time()
    time_end = round((end - start), 3)

    if -1 in search_res:
        result = [('No results were found', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley',
                   'Navec does not contain words from your query. Try another index or '
                   'change your query.', '', '', '')]
    elif -2 in search_res:
        result = [('No results were found', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley',
                   'Your query is semantically empty. Please change your query to '
                   'proceed searching.', '', '', '')]
    else:
        all_files = session.query(Texts).filter(Texts.index.in_(search_res))
        result = [(file.title, file.url, file.text, file.tags, file.date, file.topic) for file in all_files]
    return render_template('results.html', files=result, alt=time_end,
                           query=query)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
