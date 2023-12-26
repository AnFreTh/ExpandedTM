
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """class based tf_idf retrieval from cluster of documents

    Args:
        documents (_type_): _description_
        m (_type_): _description_
        ngram_range (tuple, optional): _description_. Defaults to (1, 1).

    Returns:
        _type_: _description_
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(
        documents
    )
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_tfidf_topics(tf_idf, count, docs_per_topic, n=10):
    """class based tf_idf retrieval from cluster of documents

    Args:
        tf_idf (_type_): _description_
        count (_type_): _description_
        docs_per_topic (_type_): _description_
        n (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.predictions)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {
        label: [((words[j]), (tf_idf_transposed[i][j])) for j in indices[i]][::-1]
        for i, label in enumerate(labels)
    }

    return top_n_words