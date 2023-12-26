import umap.umap_ as umap
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
from utils.topic_extraction import TopicExtractor
from utils.cleaning import clean_topics
import pandas as pd
from sklearn.mixture import GaussianMixture


data_dir = "../preprocessed_datasets"


class CEDC(AbstractModel):
    """
    A topic modeling class that utilizes sentence embeddings, UMAP for dimensionality
    reduction, and Gaussian Mixture Models (GMM) for clustering text data into topics.

    This class inherits from the AbstractModel class and is designed for clustering
    and topic extraction from textual data.

    Attributes:
        hyperparameters (dict): A dictionary of hyperparameters for the model.
        n_topics (int): The number of topics to identify in the dataset.
        embedding_model (SentenceTransformer): The sentence embedding model used to
            convert text to embeddings.
        umap_args (dict): Arguments for UMAP dimensionality reduction.
        gmm_args (dict): Arguments for the Gaussian Mixture Model.
        dataset (pandas.DataFrame): The dataset used for training, containing the text documents.
    """

    def __init__(
        self,
        hyperparameters=None,
        num_topics=20,
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_args=None,
        optim=False,
        modeltype="GMM",
    ):
        """
        Initializes the CEDC model with specified hyperparameters, number of topics,
        embedding model, UMAP arguments, and additional options.

        Parameters:
            hyperparameters (dict, optional): A dictionary containing model hyperparameters.
                Defaults to None.
            num_topics (int, optional): The number of topics to identify in the dataset.
                Defaults to 20.
            embedding_model (SentenceTransformer, optional): The model used for generating
                sentence embeddings. Defaults to SentenceTransformer("all-MiniLM-L6-v2").
            umap_args (dict, optional): A dictionary containing arguments for UMAP.
                Defaults to None.
            optim (bool, optional): If true, enables optimization. Defaults to False.
            modeltype (str, optional): Type of model to use for clustering, e.g., 'GMM'.
                Defaults to "GMM".
        """
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
        """
        Trains the CEDC model on the provided dataset, performing topic extraction
        and clustering using Gaussian Mixture Models.

        Parameters:
            dataset: The dataset to train the model on, containing text documents.
            only_nouns (bool, optional): If true, only extracts nouns for topic
                modeling. Defaults to False.
            clean (bool, optional): If true, performs cleaning of the extracted topics
                based on a similarity threshold. Defaults to False.
            clean_threshold (float, optional): The similarity threshold used for
                cleaning topics. Defaults to 0.85.
            expansion_corpus (str, optional): The name of the corpus used for topic
                expansion. Defaults to "octis".
            n_words (int, optional): The number of words to consider in each topic.
                Defaults to 20.

        Returns:
            dict: A dictionary containing the extracted topics, and potentially cleaned
            topics and centroids.
        """
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
