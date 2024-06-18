import pandas as pd
import spacy
import os
import numpy as np
import torch
import slovnet
import gensim
import re
from gensim.models import KeyedVectors
from preprocessing import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_array
from navec import Navec
from slovnet.model.emb import NavecEmbedding
from typing import Union
from transformers import AutoTokenizer, AutoModel


class BM25Matrix:
    """
    Indexation and search based on BM25 metric

    Attributes:
        corpus: list
            List of preprocessed texts
        raws: list
            List of raw texts
        nlp: spacy.lang
            Spacy model object
        vocabulary: dict
            Dict containing a word, pointer and text id
        bm25_index: np.array
            Computed index


    Methods:
        compute_index(text, k1=2.0, b=0.75):
            Computes index using BM25 formula and matrix operations
        search(query, n=5):
            Processes query and performs search
    """

    def __init__(self, texts: pd.Series, raws: pd.Series, nlp: spacy.lang) -> None:
        self.corpus = list(texts)
        self.raws = list(raws)
        self.nlp = nlp
        self.bm25_index = self.compute_index(texts)

    def compute_index(self, texts: pd.Series, k1=2.0, b=0.75) -> None:
        """
        Computes index

        Args:
            texts: pd.Series
                List of processed texts
            k1: float
                Free coefficient (default=2.0)
            b: float
                Free coefficient (default=0.75)

        Returns:
            bm25_index: np.array
                Computed index
        """
        indptr = [0]
        indices = []
        data = []
        self.vocabulary = {}
        for text in texts:
            if isinstance(text, str):
                for word in text.split():
                    freq_index = self.vocabulary.setdefault(word, len(self.vocabulary))
                    indices.append(freq_index)
                    data.append(1)
            else:
                freq_index = self.vocabulary.setdefault('', len(self.vocabulary))
                indices.append(freq_index)
                data.append(1)
            indptr.append(len(indices))

        matrix = csr_array((data, indices, indptr), dtype=int).toarray()
        N = matrix.shape[0]
        n = np.count_nonzero(matrix, axis=0)
        idf = np.log(N) - np.log(n)

        each_text_len = matrix.sum(axis=1)
        average = each_text_len.mean()
        length_norm = (1.0 - b + b * (each_text_len / average)).reshape(-1, 1)
        modified_tf = matrix * ((k1 + 1.0) / (k1 * length_norm + matrix))
        self.bm25_index = idf * modified_tf
        np.save('./data/bm25.npy', self.bm25_index)

        return self.bm25_index

    def search(self, query: str, n=5) -> list:
        """
        Performs search on a given query

        Args:
            query: str
                Query to base search on
            n: int
                Size of search result (default=5)

        Returns:
            vector_final: list
                List of matching text identifiers
        """
        query = preprocess_text(query, self.nlp)[0].split()
        if query == []:
            return [-2]
        indices = []
        for word in query:
            if word in self.vocabulary.keys():
                indices.append(self.vocabulary[word])

        word_num = len(self.vocabulary)
        query_index = np.zeros((1, word_num))
        query_index[0, indices] = 1
        query_index = query_index.transpose()

        result = -self.bm25_index.dot(query_index)
        result = result.argsort(axis=0).tolist()
        if len(result) >= n:
            result = result[:n]

        vector_final = [int(idx[0]) for idx in result]
        return vector_final


class Word2vecIndex:
    """
        Indexation and search based on Word2Vec

        Attributes:
            corpus: list
                List of preprocessed texts
            raws: list
                List of raw texts
            nlp: spacy.lang
                Spacy model object
            model: KeyedVectors
                Word2Vec model object

        Methods:
            compute_index():
                Computes index using Word2Vec vectorization
            search(query, n=5):
                Processes query and performs search
        """

    def __init__(self, texts: pd.Series, raw_texts: pd.Series,
                 w2v: KeyedVectors, nlp: spacy.lang) -> None:
        self.corpus = list(texts)
        self.raws = list(raw_texts)
        self.nlp = nlp
        self.model = w2v

        if os.path.exists('data/w2v.npy'):
            with open('data/w2v.npy', 'rb') as f:
                self.index = np.load(f)
        else:
            self.index = self.compute_index()

    def compute_index(self):
        """
        Computes index

        Returns:
            text_embeddings: np.array
                Embeddings computed for each text in corpus
        """
        text_embeddings = np.zeros((len(self.corpus), 300))
        for text_id in range(len(self.corpus)):
            try:
                self.corpus[text_id] = self.corpus[text_id].split()
                text_embeddings[text_id] = self.model.get_mean_vector(self.corpus[text_id])
            except:
                text_embeddings[text_id] = np.zeros((1, 300))
        np.save('data/w2v.npy', text_embeddings)

        return text_embeddings

    def search(self, query: str, n=5) -> list:
        """
        Performs search on a given query

        Args:
            query: str
                Query to base search on
            n: int
                Size of search result (default=5)

        Returns:
            vector_final: list
                List of matching text identifiers
        """
        query = preprocess_text(query, self.nlp)[1].split()
        print(query)
        if query == []:
            return [-2]
        query = self.model.get_mean_vector(query).reshape(1, -1)
        vector = cosine_similarity(self.index, query)
        vector = np.argsort(-vector, axis=0)
        if len(vector) >= n:
            vector = vector[:n]

        vector_final = [int(idx[0]) for idx in vector]
        return vector_final


class NavecIndex:
    """
    Indexation and search based on Navec

    Attributes:
        corpus: list
            List of preprocessed texts
        raws: list
            List of raw texts
        nlp: spacy.lang
            Spacy model object
        navec: Navec
            Navec model object

    Methods:
        compute_index():
            Computes index using Word2Vec vectorization
        search(query, n=5):
            Processes query and performs search
    """

    def __init__(self, texts: pd.Series, raw_texts: pd.Series, navec: Navec,
                 nlp: spacy.lang) -> None:
        self.corpus = list(texts)
        self.raws = list(raw_texts)
        self.nlp = nlp

        self.navec = navec
        self.emb = NavecEmbedding(self.navec)

        if os.path.exists('data/navec_index.npy'):
            with open('data/navec_index.npy', 'rb') as f:
                self.index = np.load(f)
        else:
            self.index = self.compute_index()

    def compute_index(self):
        """
        Computes index

        Returns:
           text_embeddings: np.array
               Embeddings computed for each text in corpus
        """

        text_embeddings = np.zeros((len(self.corpus), 300))
        for text_id in range(len(self.corpus)):
            try:
                self.corpus[text_id] = self.corpus[text_id].split()
                ids = [self.navec.vocab[i] for i in self.corpus[text_id] if i in self.navec]
                input = torch.tensor(ids)
                output = self.emb(input)
                single_embedding = torch.mean(output, 0)
                text_embeddings[text_id] = single_embedding
            except:
                text_embeddings[text_id] = np.zeros((1, 300))

        np.save('data/navec_index.npy', text_embeddings)

        return text_embeddings

    def search(self, query: str, n=5) -> Union[list, str]:
        """
        Performs search on a given query

        Args:
            query: str
                Query to base search on
            n: int
                Size of search result (default=5)

        Returns:
            vector_final: list
                List of matching text identifiers
        """

        try:
            query = preprocess_text(query, self.nlp)[0].split()
            if query == []:
                return [-2]
            ids = [self.navec.vocab[i] for i in query if i in self.navec]
            output = self.emb(torch.tensor(ids))
            final_query = torch.mean(output, 0).reshape(1, -1)

            vector = cosine_similarity(self.index, final_query)
            vector = np.argsort(-vector, axis=0)
            if len(vector) >= n:
                vector = vector[:n]

            vector_final = [int(idx[0]) for idx in vector]
            return vector_final
        except:
            return [-1]


class BertIndex:
    """
    Indexation and search based on Bert-like transformer

    Attributes:
        raws: list
            List of raw texts
        tokenizer: AutoTokenizer
            Tokenizer object that is necessary for Bert functioning
        bert: AutoModel
            Bert model object

    Methods:
        compute_index():
            Computes index using Word2Vec vectorization
        search(query, n=5):
            Processes query and performs search
    """

    def __init__(self, raws: pd.Series, tokenizer: AutoTokenizer, bert: AutoModel) -> None:
        self.corpus = list(raws)
        self.tokenizer = tokenizer
        self.bert = bert
        if os.path.exists('data/bert_index.npy'):
            with open('data/bert_index.npy', 'rb') as f:
                self.index = np.load(f)
        else:
            self.index = self.compute_index()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        A mean-pool layer compression
        """

        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def compute_index(self):
        """
        Computes index

        Returns:
           text_embeddings: np.array
               Embeddings computed for each text in corpus
        """

        for text in range(len(self.corpus)):
            self.corpus[text] = self.corpus[text].split()[:512]
            if len(self.corpus[text]) > 512:
                self.corpus[text] = self.corpus[text][:512]
            self.corpus[text] = ' '.join(self.corpus[text])
        encoded_input = self.tokenizer(self.corpus, padding=True, truncation=True,
                                       max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = self.bert(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        np.save('data/bert_index.npy', sentence_embeddings)
        return sentence_embeddings

    def search(self, query, n=5):
        """
        Performs search on a given query

        Args:
            query: str
                Query to base search on
            n: int
                Size of search result (default=5)

        Returns:
            vector_final: list
                List of matching text identifiers
        """
        query = self.tokenizer(query, padding=True, truncation=True,
                               max_length=24, return_tensors='pt')
        if query == []:
            return [-2]
        with torch.no_grad():
            model_output = self.bert(**query)
        final_query = self.mean_pooling(model_output, query['attention_mask'])

        vector = cosine_similarity(self.index, final_query)
        vector = np.argsort(-vector, axis=0)
        if len(vector) >= n:
            vector = vector[:n]

        vector_final = [int(idx[0]) for idx in vector]
        return vector_final


class AllModels:
    """
    Gathering everything together
    """

    def __init__(self,
                 texts: pd.Series,
                 texts_pos: pd.Series,
                 raw_texts: pd.Series,
                 navec: Navec,
                 word2v: KeyedVectors,
                 tokenizer: AutoTokenizer,
                 bert: AutoModel,
                 nlp: spacy.lang) -> None:
        self.bm25_i = BM25Matrix(texts, raw_texts, nlp)
        self.word2v_i = Word2vecIndex(texts_pos, raw_texts, word2v, nlp)
        self.navec_i = NavecIndex(texts, raw_texts, navec, nlp)
        self.bert_i = BertIndex(raw_texts, tokenizer, bert)
