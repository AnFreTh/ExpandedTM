import umap.umap_ as umap
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
from utils.topic_extraction import TopicExtractor
from utils.cleaning import clean_topics
import pandas as pd
from sklearn.mixture import GaussianMixture


data_dir = "../preprocessed_datasets"


class CEDC(AbstractModel):
    def __init__(
        self,
        hyperparameters=None,
        num_topics=20,
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_args=None,
        optim=False,
        modeltype="GMM",
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
                "n_components": 8,
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

        self.gmm_args = {
            "n_components": self.n_topics,
            "covariance_type": "full",
            "tol": 0.001,
            "reg_covar": 0.000001,
            "max_iter": 100,
            "n_init": 1,
            "init_params": "kmeans",
        }

    def train_model(
        self,
        dataset,
        only_nouns=False,
        clean=False,
        clean_threshold=0.85,
        expansion_corpus="octis",
        n_words=20,
    ):
        reducer = umap.UMAP(**self.umap_args)
        self.dataset = dataset
        gmm_data = pd.DataFrame()
        gmm_data["tokens"] = self.dataset.get_corpus()
        gmm_data["text"] = [" ".join(words) for words in gmm_data["tokens"]]
        gmm_data["label_text"] = self.dataset.get_labels()
        gmm_data["docs"] = gmm_data["text"]

        print("----------------------- encoding the documents -----------------------")
        self.embeddings = self.embedding_model.encode(gmm_data["text"])

        print("----------------------- reduce embedding dimensions ------------------")
        reduced_embeddings = reducer.fit_transform(self.embeddings)

        print("----------------------- Cluster documents ----------------------------")
        self.GMM = GaussianMixture(
            **self.gmm_args,
        ).fit(reduced_embeddings)

        gmm_predictions = self.GMM.predict_proba(reduced_embeddings)

        predictions = pd.DataFrame(gmm_predictions)

        TE = TopicExtractor(
            dataset=self.dataset,
            topic_assignments=predictions,
            n_topics=self.n_topics,
            embedding_model=self.embedding_model,
        )

        print("----------------------- extract topics -------------------------------")
        topics, topic_centroids = TE._noun_extractor_haystack(
            self.embeddings,
            n=n_words + 20,
            corpus=expansion_corpus,
            only_nouns=only_nouns,
        )

        if clean:
            cleaned_topics, cleaned_centroids = clean_topics(
                topics, similarity=clean_threshold, embedding_model=self.embedding_model
            )
            topics = cleaned_topics
            topic_centroids = cleaned_centroids

        words_list = []
        new_topics = {}
        for k in range(self.n_topics):
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
        res_dic["topic-word-matrix"] = None

        return res_dic
