from octis.evaluation_metrics.metrics import AbstractMetric
from sentence_transformers import SentenceTransformer
from _helper_funcs import (
    Embed_corpus,
    Embed_topic,
    Update_corpus_dic_list,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ISIM(AbstractMetric):
    """
    For each topic, draw several intruder words that are not from the same topic by first selecting some topics that are not the specific topic and
    then selecting one word from each of those topics.
    The intruder score for the topic is then calculated as the average cosine similarity of the intruder words and the top words.
    """

    def __init__(
        self,
        dataset,
        n_intruders=1,
        n_words=10,
        metric_embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
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
            metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )
        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                metric_embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )

        self.n_intruders = n_intruders
        self.corpus_dict = tw_emb
        self.n_words = n_words
        self.embeddings = None

    def score_one_intr_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculate the score for each topic but only with one intruder word
        """
        if new_Embeddings:  # for this function, reuse embeddings per default
            self.embeddings = None

        topics_tw = model_output["topics"]

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

        avg_sim_topic_list = (
            []
        )  # iterate over each topic and append the average similarity to the intruder word
        for idx, topic in enumerate(emb_tw):
            mask = np.full(emb_tw.shape[0], True)  # mask out the current topic
            mask[idx] = False

            other_topics = emb_tw[
                mask
            ]  # embeddings of every other topic except the current one

            intr_topic_idx = np.random.randint(
                other_topics.shape[0]
            )  # select random topic index
            intr_word_idx = np.random.randint(
                other_topics.shape[1]
            )  # select random word index

            intr_embedding = other_topics[
                intr_topic_idx, intr_word_idx
            ]  # select random word

            sim = cosine_similarity(
                intr_embedding.reshape(1, -1), topic
            )  # calculate all pairwise similarities of intruder words and top words

            avg_sim_topic_list.append(np.mean(sim))

        return np.array(avg_sim_topic_list)

    def score_one_intr(self, model_output, new_Embeddings=True):
        """
        Calculate the score for all topics combined but only with one intruder word
        """
        if new_Embeddings:
            self.embeddings = None
        return np.mean(self.score_one_intr_per_topic(model_output, new_Embeddings))

    def score_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculate the score for each topic with several intruder words
        """
        if new_Embeddings:
            self.embeddings = None
        score_lis = []
        for _ in range(self.n_intruders):  # iterate over the number of intruder words
            score_per_topic = self.score_one_intr_per_topic(
                model_output, new_Embeddings=False
            )  # calculate the intruder score, but re-use embeddings
            score_lis.append(score_per_topic)  # and append to list

        res = np.vstack(
            score_lis
        ).T  # stack all scores and transpose to get a (n_topics, n_intruder words) matrix

        self.embeddings = None
        return np.mean(res, axis=1)  # return the mean score for each topic

    def score(self, model_output, new_Embeddings=True):
        if new_Embeddings:
            self.embeddings = None
        """
        Calculate the score for all topics combined but only with several intruder words
        """
        return float(np.mean(self.score_per_topic(model_output)))


class INT(AbstractMetric):
    """
    For each topic, draw several intruder words that are not from the same topic by first selecting some topics that are not the specific topic and
    then selecting one word from each of those topics.
    The embedding intruder cosine similarity accuracy for one intruder word is then calculated by the fraction of top words
    that are least similar to the intruder
    """

    def __init__(
        self,
        dataset,
        metric_embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
        n_intruders=1,
        n_words=10,
    ):
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider
        """

        tw_emb = Embed_corpus(
            dataset,
            metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )
        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                metric_embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )

        self.n_intruders = n_intruders
        self.corpus_dict = tw_emb
        self.n_words = n_words
        self.embeddings = None

    def score_one_intr_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculate the score for each topic but only with one intruder word
        """
        if new_Embeddings:
            self.embeddings = None
        topics_tw = model_output["topics"]

        if self.embeddings is None:
            emb_tw = Embed_topic(
                topics_tw, self.corpus_dict, self.n_words
            )  # embed the top words
            emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
                :, : self.n_words, :
            ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)
            self.embeddings = emb_tw
        else:
            emb_tw = (
                self.embeddings
            )  # create tensor of size (n_topics, n_topwords, n_embedding_dims)

        avg_sim_topic_list = []
        for idx, topic in enumerate(emb_tw):
            mask = np.full(emb_tw.shape[0], True)  # mask out the current topic
            mask[idx] = False

            other_topics = emb_tw[
                mask
            ]  # embeddings of every other topic except the current one

            intr_topic_idx = np.random.randint(
                other_topics.shape[0]
            )  # select random topic index
            intr_word_idx = np.random.randint(
                other_topics.shape[1]
            )  # select random word index

            intr_embedding = other_topics[
                intr_topic_idx, intr_word_idx
            ]  # select random word

            new_words = np.vstack(
                [intr_embedding, topic]
            )  # stack the intruder embedding above the other embeddings to get a matrix with shape ((1+n_topwords), n_embedding_dims)

            sim = cosine_similarity(
                new_words
            )  # calculate all pairwise similarities for matrix of shape ((1+n_topwords, 1+n_topwords))

            least_similar = np.argmin(
                sim[1:], axis=1
            )  # for each word, except the intruder, calculate the index of the least similar word
            intr_acc = np.mean(
                least_similar == 0
            )  # calculate the fraction of words for which the least similar word is the intruder word (at index 0)

            avg_sim_topic_list.append(
                intr_acc
            )  # append intruder accuracy for this sample

        return np.array(avg_sim_topic_list)

    def score_one_intr(self, model_output, new_Embeddings=True):
        if new_Embeddings:
            self.embeddings = None
        self.embeddings = None
        """
        Calculate the score for all topics combined but only with one intruder word
        """
        return np.mean(self.score_one_intr_per_topic(model_output))

    def score_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculate the score for each topic with several intruder words
        """
        if new_Embeddings:
            self.embeddings = None

        score_lis = []
        for _ in range(self.n_intruders):
            score_per_topic = self.score_one_intr_per_topic(
                model_output, new_Embeddings=False
            )
            score_lis.append(score_per_topic)
        self.embeddings = None
        res = np.vstack(score_lis).T

        return np.mean(res, axis=1)

    def score(self, model_output, new_Embeddings=True):
        if new_Embeddings:
            self.embeddings = None
        """
        Calculate the score for all topics combined but only with several intruder words
        """
        self.embeddings = None
        return float(np.mean(self.score_per_topic(model_output)))