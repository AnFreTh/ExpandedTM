from sklearn.cluster import KMeans
import umap.umap_ as umap
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from utils.tf_idf import c_tf_idf, extract_tfidf_topics



class KmeansTM(AbstractModel):
    def __init__(
        self,
        hyperparameters=None,
        num_topics=20,
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_args=None,
    ):
        super().__init__()

        if hyperparameters == None:
            self.hyperparameters = {}
        else:
            self.hyperparameters = hyperparameters
        self.n_topics = num_topics
        self.embedding_model = embedding_model
        if umap_args is None:
            self.umap_args = {
                "n_neighbors": 15,
                "n_components": 15,
                "metric": "cosine",
                "n_epochs": 1000,
                "learning_rate": 1.0,
                "init": "random",
                "min_dist": 0.1,
                "spread": 1.0,
                "low_memory": False,
                "set_op_mix_ratio": 1.0,
                "local_connectivity": 1,
                "repulsion_strength": 1.0,
                "negative_sample_rate": 5,
                "transform_queue_size": 4.0,
                "a": None,
                "b": None,
                "random_state": 101,
                "metric_kwds": None,
                "angular_rp_forest": False,
                "target_n_neighbors": -1,
                "target_metric": "categorical",
                "target_metric_kwds": None,
                "target_weight": 0.5,
                "transform_seed": 42,
                "verbose": False,
                "unique": False,
            }
        else:
            self.umap_args = umap_args

    def train_model(self, dataset):
        reducer = umap.UMAP(**self.umap_args)
        self.dataset = dataset
        kmeans_data = pd.DataFrame()
        kmeans_data["tokens"] = self.dataset.get_corpus()
        kmeans_data["text"] = [" ".join(words) for words in kmeans_data["tokens"]]
        kmeans_data["label_text"] = self.dataset.get_labels()
        kmeans_data["docs"] = kmeans_data["text"]
        self.dataset = kmeans_data
        embedded_docs = self.embedding_model.encode(self.dataset["text"])

        reduced_embeddings = reducer.fit_transform(embedded_docs)
        clustering_model = KMeans(n_clusters=20)
        clustering_model.fit(reduced_embeddings)
        self.dataset["predictions"] = pd.DataFrame(clustering_model.labels_)

        docs_per_topic = self.dataset.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )
        tfidf, count = c_tf_idf(docs_per_topic["text"].values, m=len(self.dataset))
        topics = extract_tfidf_topics(
            tfidf,
            count,
            docs_per_topic,
            n=10,
        )

        new_topics = {}
        words_list = []
        for k in self.dataset["predictions"].unique():
            words = [
                word for t in topics[k][0:10] for word in t if isinstance(word, str)
            ]
            weights = [
                weight
                for t in topics[k][0:10]
                for weight in t
                if isinstance(weight, float)
            ]
            weights = [weight / sum(weights) for weight in weights]
            new_topics[k] = list(zip(words, weights))
            words_list.append(words)

        res_dic = {}
        res_dic["topics"] = words_list
        res_dic["topic-word-matrix"] = tfidf.T

        return res_dic