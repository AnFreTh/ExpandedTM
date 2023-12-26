
from octis.evaluation_metrics.metrics import AbstractMetric
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from octis.evaluation_metrics._helper_funcs import (
    cos_sim_pw,
    Embed_topic,
    Embed_corpus,
    Update_corpus_dic_list,
    Embed_stopwords,
)
from sentence_transformers import SentenceTransformer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim

gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS
nltk_stopwords = stopwords.words("english")
stopwords = list(
    set(nltk_stopwords + list(gensim_stopwords) + list(ENGLISH_STOP_WORDS))
)


class Embedding_Topic_Diversity(AbstractMetric):
    """
    Measure the diversity of the topics by calculating the mean cosine similarity
    of the mean vectors of the top words of all topics
    """

    def __init__(
        self,
        dataset,
        n_words=10,
        embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
    ):
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider
        """

        tw_emb = Embed_corpus(
            dataset,
            embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )

        self.n_words = n_words
        self.corpus_dict = tw_emb

    def score(self, model_output):
        topics_tw = model_output["topics"]  # size: (n_topics, voc_size)
        topic_weights = model_output["topic-word-matrix"][
            :, : self.n_words
        ]  # select the weights of the top words

        topic_weights = topic_weights / np.sum(topic_weights, axis=1).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        emb_tw = Embed_topic(
            topics_tw, self.corpus_dict, self.n_words
        )  # embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
            :, : self.n_words, :
        ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)

        weighted_vecs = (
            topic_weights[:, :, None] * emb_tw
        )  # multiply each embedding vector with its corresponding weight
        topic_means = np.sum(
            weighted_vecs, axis=1
        )  # calculate the sum, which yields the weighted average

        return float(cos_sim_pw(topic_means))

    def score_per_topic(self, model_output):
        topics_tw = model_output["topics"]  # size: (n_topics, voc_size)
        topic_weights = model_output["topic-word-matrix"][
            :, : self.n_words
        ]  # select the weights of the top words size: (n_topics, n_topwords)

        topic_weights = topic_weights / np.sum(topic_weights, axis=1).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        emb_tw = Embed_topic(
            topics_tw, self.corpus_dict, self.n_words
        )  # embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
            :, : self.n_words, :
        ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)
        self.embeddings = emb_tw

        weighted_vecs = (
            topic_weights[:, :, None] * emb_tw
        )  # multiply each embedding vector with its corresponding weight
        topic_means = np.sum(
            weighted_vecs, axis=1
        )  # calculate the sum, which yields the weighted average

        sim = cosine_similarity(
            topic_means
        )  # calculate the pairwise cosine similarity of the topic means
        sim_mean = (np.sum(sim, axis=1) - 1) / (
            len(sim) - 1
        )  # average the similarity of each topic's mean to the mean of every other topic

        return sim_mean


class Expressivity(AbstractMetric):
    """
    Measure the distance of the mean of the topic topwords to the mean of the embedding of the stop words
    """

    def __init__(
        self,
        dataset,
        stopword_list=stopwords,
        n_words=10,
        embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
    ):
        """
        stopword_corpus: The list of all stopwords to compare with; i.e. the
        specific stopwords of this corpus
        """

        tw_emb = Embed_corpus(
            dataset,
            embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )
        self.stopword_list = stopword_list

        self.n_words = n_words
        self.corpus_dict = tw_emb
        self.embeddings = None

        self.stopword_emb = Embed_stopwords(
            stopword_list, embedder
        )  # embed all the stopwords size: (n_stopwords, emb_dim)
        self.stopword_mean = np.mean(
            np.array(self.stopword_emb), axis=0
        )  # mean of stopword embeddings

    def score(self, model_output, new_Embeddings=True):
        if new_Embeddings:
            self.embeddings = None
        return float(np.mean(self.score_per_topic(model_output, new_Embeddings)))

    def score_per_topic(self, model_output, new_Embeddings=True):
        if new_Embeddings:
            self.embeddings = None

        topics_tw = model_output["topics"]  # size: (n_topics, voc_size)
        topic_weights = model_output["topic-word-matrix"][
            :, : self.n_words
        ]  # select the weights of the top words

        topic_weights = topic_weights / np.sum(topic_weights, axis=1).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        if self.embeddings is None:
            emb_tw = Embed_topic(
                topics_tw, self.corpus_dict, self.n_words
            )  # embed the top words
            emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
                :, : self.n_words, :
            ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)
            self.embeddings = emb_tw
        else:
            emb_tw = self.embeddings

        weighted_vecs = (
            topic_weights[:, :, None] * emb_tw
        )  # multiply each embedding vector with its corresponding weight
        topic_means = np.sum(
            weighted_vecs, axis=1
        )  # calculate the sum, which yields the weighted average

        if np.isnan(topic_means.sum()) != 0:
            raise ValueError("There are some nans in the topic means")

        topword_sims = []
        # iterate over every topic and append the cosine similarity of the topic's centroid and the stopword mean
        for mean in topic_means:
            topword_sims.append(
                cosine_similarity(
                    mean.reshape(1, -1), self.stopword_mean.reshape(1, -1)
                )[0, 0]
            )

        return np.array(topword_sims)