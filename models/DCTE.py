from setfit import SetFitModel, SetFitTrainer
import pyarrow as pa
import pandas as pd
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from octis.models.model import AbstractModel
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from utils.tf_idf import c_tf_idf, extract_tfidf_topics
from sklearn.model_selection import train_test_split


class DCTE(AbstractModel):
    """
    A document classification and topic extraction class that utilizes the SetFitModel for
    document classification and TF-IDF for topic extraction.

    This class inherits from the AbstractModel class and is designed for supervised
    document classification and unsupervised topic modeling.

    Attributes:
        n_topics (int): The number of topics to identify in the dataset.
        embedding_model (SentenceTransformer): The sentence embedding model used to
            convert text to embeddings.
        model (SetFitModel): The SetFitModel used for document classification.
        batch_size (int): The batch size used in training.
        num_iterations (int): The number of iterations for SetFit training.
        num_epochs (int): The number of epochs for SetFit training.
    """

    def __init__(
        self,
        num_topics=20,
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        model="all-MiniLM-L6-v2",
        batch_size=8,
        num_iterations=20,
        num_epochs=10,
    ):
        """
        Initializes the DCTE model with specified number of topics, embedding model,
        SetFit model, batch size, number of iterations, and number of epochs.

        Parameters:
            num_topics (int, optional): The number of topics to identify in the dataset.
                Defaults to 20.
            embedding_model (SentenceTransformer, optional): The model used for generating
                sentence embeddings. Defaults to SentenceTransformer("all-MiniLM-L6-v2").
            model (str, optional): The identifier of the SetFit model to be used.
                Defaults to "all-MiniLM-L6-v2".
            batch_size (int, optional): The batch size to use during training. Defaults to 8.
            num_iterations (int, optional): The number of iterations to run SetFit training.
                Defaults to 20.
            num_epochs (int, optional): The number of epochs to train SetFit model.
                Defaults to 10.
        """
        self.n_topics = num_topics
        self.embedding_model = embedding_model
        self.model = SetFitModel.from_pretrained(f"sentence-transformers/{model}")

        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs

    def train_model(self, train_dataset, predict_dataset, val_split=0.2, top_words=10):
        """
        Trains the DCTE model using the given training dataset and then performs
        prediction and topic extraction on the specified prediction dataset.

        The method uses the SetFitTrainer for training and evaluates the model's performance.
        It then applies the trained model for prediction and extracts topics using TF-IDF.

        Parameters:
            train_dataset: The dataset used for training the model.
            predict_dataset: The dataset on which to perform prediction and topic extraction.
            val_split (float, optional): The fraction of the training data to use as
                validation data. Defaults to 0.2.
            top_words (int, optional): The number of top words to extract for each topic.
                Defaults to 10.

        Returns:
            dict: A dictionary containing the extracted topics and the topic-word matrix.
        """
        df = pd.DataFrame()

        df["tokens"] = train_dataset.get_corpus()
        df["text"] = [" ".join(words) for words in df["tokens"]]
        df["label_text"] = train_dataset.get_labels()
        df["label"] = preprocessing.LabelEncoder().fit_transform(df["label_text"])

        train_df, val_df = train_test_split(df, test_size=val_split)

        ### convert to Huggingface dataset
        self.train_ds = Dataset(pa.Table.from_pandas(train_df))
        self.val_ds = Dataset(pa.Table.from_pandas(val_df))

        self.trainer = SetFitTrainer(
            model=self.model,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            loss_class=CosineSimilarityLoss,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations,
            num_epochs=self.num_epochs,
        )

        # train
        self.trainer.train()
        # evaluate accuracy
        metrics = self.trainer.evaluate()

        print("################ finished training #############")
        print(metrics)

        predict_df = pd.DataFrame
        predict_df["tokens"] = predict_dataset.get_corpus()
        predict_df["text"] = [" ".join(words) for words in predict_df["tokens"]]

        predict_df["predictions"] = self.model(predict_df["text"])

        docs_per_topic = predict_df.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )
        tfidf, count = c_tf_idf(docs_per_topic["text"].values, m=len(predict_df))
        topics = extract_tfidf_topics(
            tfidf,
            count,
            docs_per_topic,
            n=top_words,
        )

        new_topics = {}
        words_list = []
        for k in predict_df["predictions"].unique():
            words = [
                word
                for t in topics[k][0:top_words]
                for word in t
                if isinstance(word, str)
            ]
            weights = [
                weight
                for t in topics[k][0:top_words]
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
