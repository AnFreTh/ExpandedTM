import numpy as np
import nltk
import re
from octis.evaluation_metrics.metrics import AbstractMetric
from _helper_funcs import (
    cos_sim_pw,
    Embed_corpus,
    Embed_topic,
    Update_corpus_dic_list,
)
from octis.dataset.dataset import Dataset
from sentence_transformers import SentenceTransformer
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim

gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS


nltk_stopwords = stopwords.words("english")

stopwords = list(
    set(nltk_stopwords + list(gensim_stopwords) + list(ENGLISH_STOP_WORDS))
)


class NPMI(AbstractMetric):
    def __init__(
        self,
        dataset,
        stopwords=stopwords,
        n_topics=20,
    ):
        self.stopwords = stopwords
        self.ntopics = n_topics
        self.dataset = dataset

        files = self.dataset.get_corpus()
        self.files = [" ".join(words) for words in files]

    def _create_vocab_preprocess(self, data, preprocess, process_data=False):
        word_to_file = {}
        word_to_file_mult = {}

        process_files = []
        for file_num in range(0, len(data)):
            words = data[file_num].lower()
            words = words.strip()
            words = re.sub("[^a-zA-Z0-9]+\s*", " ", words)
            words = re.sub(" +", " ", words)
            # .translate(strip_punct).translate(strip_digit)
            words = words.split()
            # words = [w.strip() for w in words]
            proc_file = []

            for word in words:
                if word in self.stopwords or word == "dlrs" or word == "revs":
                    continue
                if word in word_to_file:
                    word_to_file[word].add(file_num)
                    word_to_file_mult[word].append(file_num)
                else:
                    word_to_file[word] = set()
                    word_to_file_mult[word] = []

                    word_to_file[word].add(file_num)
                    word_to_file_mult[word].append(file_num)

            process_files.append(proc_file)

        for word in list(word_to_file):
            if len(word_to_file[word]) <= preprocess or len(word) <= 3:
                word_to_file.pop(word, None)
                word_to_file_mult.pop(word, None)

        if process_data:
            vocab = word_to_file.keys()
            files = []
            for proc_file in process_files:
                fil = []
                for w in proc_file:
                    if w in vocab:
                        fil.append(w)
                files.append(" ".join(fil))

            data = files

        return word_to_file, word_to_file_mult, data

    def _create_vocab_and_files(self, preprocess=5):
        return self._create_vocab_preprocess(self.files, preprocess)

    def score(self, model_output):
        topic_words = model_output["topics"]
        (
            word_doc_counts,
            dev_word_to_file_mult,
            dev_files,
        ) = self._create_vocab_and_files(preprocess=1)
        nfiles = len(dev_files)
        eps = 10 ** (-12)

        all_topics = []
        for k in range(self.ntopics):
            topic_score = []

            ntopw = len(topic_words[k])

            for i in range(ntopw - 1):
                for j in range(i + 1, ntopw):
                    w1 = topic_words[k][i]
                    w2 = topic_words[k][j]

                    w1w2_dc = len(
                        word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set())
                    )
                    w1_dc = len(word_doc_counts.get(w1, set()))
                    w2_dc = len(word_doc_counts.get(w2, set()))

                    # Correct eps:
                    pmi_w1w2 = np.log(
                        (w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps
                    )
                    npmi_w1w2 = pmi_w1w2 / (-np.log((w1w2_dc) / nfiles + eps))

                    topic_score.append(npmi_w1w2)

            all_topics.append(np.mean(topic_score))

        avg_score = np.around(np.mean(all_topics), 5)

        return avg_score

    def npmi_per_topic(self, topic_words, ntopics, preprocess=5):
        (
            word_doc_counts,
            dev_word_to_file_mult,
            dev_files,
        ) = self._create_vocab_and_files(dataset=self.dataset, preprocess=preprocess)
        nfiles = len(dev_files)
        eps = 10 ** (-12)

        all_topics = []
        for k in range(ntopics):
            topic_score = []

            ntopw = len(topic_words[k])

            for i in range(ntopw - 1):
                for j in range(i + 1, ntopw):
                    w1 = topic_words[k][i]
                    w2 = topic_words[k][j]

                    w1w2_dc = len(
                        word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set())
                    )
                    w1_dc = len(word_doc_counts.get(w1, set()))
                    w2_dc = len(word_doc_counts.get(w2, set()))

                    # Correct eps:
                    pmi_w1w2 = np.log(
                        (w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps
                    )
                    npmi_w1w2 = pmi_w1w2 / (-np.log((w1w2_dc) / nfiles + eps))

                    topic_score.append(npmi_w1w2)

            all_topics.append(np.mean(topic_score))

        results = {}
        for k in range(ntopics):
            results[", ".join(topic_words[k])] = np.around(all_topics[k], 5)

        return results



class Embedding_Coherence(AbstractMetric):
    """
    Average cosine similarity between all top words in a topic
    """

    def __init__(
        self,
        dataset,
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

        self.n_words = n_words
        self.corpus_dict = tw_emb
        self.embeddings = None

    def score_per_topic(self, model_output):
        topics_tw = model_output["topics"]

        emb_tw = Embed_topic(
            topics_tw, self.corpus_dict, self.n_words
        )  # embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
            :, : self.n_words, :
        ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)
        self.embeddings = emb_tw

        topic_sims = []
        for (
            topic_emb
        ) in (
            emb_tw
        ):  # for each topic append the average pairwise cosine similarity within its words
            topic_sims.append(cos_sim_pw(topic_emb))

        return np.array(topic_sims)

    def score(self, model_output):
        res = self.score_per_topic(model_output)
        return sum(res) / len(res)


def _load_default_texts():
    """
    Loads default general texts

    Returns
    -------
    result : default 20newsgroup texts
    """
    dataset = Dataset()
    dataset.fetch_dataset("20NewsGroup")
    return dataset.get_corpus()