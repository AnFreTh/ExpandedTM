from octis.dataset.dataset import Dataset as OCTISDataset
import os
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from octis.preprocessing.preprocessing import Preprocessing
import numpy as np
import os
import pickle


class TMDataset(OCTISDataset):
    def __init__(self):
        """
        Initializes a Topic Model (TM) Dataset, which is a subclass of OCTISDataset.

        The dataset_registry attribute contains a list of supported datasets.
        """
        super().__init__()

        self.dataset_registry = [
            "20NewsGroup",
            "M10",
            "Spotify",
            "Spotify_most_popular",
            "Poliblogs",
            "Reuters",
            "BBC_News",
            "DBLP",
            "DBPedia_IT",
            "Europarl_IT",
        ]

    def fetch_dataset(self, name, dataset_path=None):
        # Check if the path is relative to the package datasets
        self.name = name
        if dataset_path is None:
            dataset_path = self.get_package_dataset_path(name)

        if os.path.exists(dataset_path):
            # If the dataset exists in the package, load it from there
            super().load_custom_dataset_from_folder(dataset_path)
        else:
            # Otherwise, load it from the given path
            super().fetch_dataset(name)

    def get_dataframe(self):
        self.dataframe = pd.DataFrame(
            {
                "tokens": self.get_corpus(),
                "label_text": self.get_labels(),
            }
        )
        self.dataframe["text"] = [" ".join(words) for words in self.dataframe["tokens"]]

    @staticmethod
    def get_package_dataset_path(name):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the 'MyPackage' directory
        my_package_dir = os.path.dirname(script_dir)
        # Construct the path to the 'preprocessed_datasets' within 'MyPackage'
        dataset_path = os.path.join(my_package_dir, "preprocessed_datasets", name)
        return dataset_path

    @staticmethod
    def get_package_embeddings_path(name):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the 'MyPackage' directory
        my_package_dir = os.path.dirname(script_dir)
        # Construct the path to the 'preprocessed_datasets' within 'MyPackage'
        dataset_path = os.path.join(my_package_dir, "pre_embedded_datasets", name)
        return dataset_path

    def get_embeddings(
        self, embedding_model_name, path: str = None, file_name: str = None
    ):
        if self.name not in self.dataset_registry and path is None:
            raise ValueError(
                "Please specify a dataset path and a file path where to save the embedding files"
            )
        # Construct the dataset folder path
        if path is not None:
            dataset_folder = path
        else:
            dataset_folder = self.get_package_embeddings_path(self.name)

        # Ensure the dataset folder exists or create it if it doesn't
        os.makedirs(dataset_folder, exist_ok=True)

        if file_name is not None:
            # Construct the embeddings file path
            embeddings_file = os.path.join(
                dataset_folder,
                file_name,
            )
        else:
            # Construct the embeddings file path
            embeddings_file = os.path.join(
                dataset_folder,
                f"{self.name}_embeddings_{embedding_model_name}.pkl",
            )

        self.get_dataframe()

        if os.path.exists(embeddings_file):
            # Load existing embeddings
            print("--- loading pre-computed embeddings ---")
            with open(embeddings_file, "rb") as file:
                embeddings = pickle.load(file)
        else:
            # Generate and save embeddings
            print("--- Create Embeddings ---")
            embeddings = self._generate_embeddings(embedding_model_name)
            with open(embeddings_file, "wb") as file:
                pickle.dump(embeddings, file)

        return embeddings

    def _generate_embeddings(self, embedding_model_name):
        # Generate embeddings
        self.embedding_model = SentenceTransformer(embedding_model_name)
        embeddings = self.embedding_model.encode(
            self.dataframe["text"], show_progress_bar=False
        )

        return embeddings

    @staticmethod
    def clean_text(text):
        # Your cleaning logic
        text = text.replace("\n", " ").replace("\r", " ").replace("\\", "")
        text = re.sub(r"[{}[\]-]", "", text)
        text = text.encode("utf-8", "replace").decode("utf-8")
        return text

    def create_load_save_dataset(
        self,
        data,
        dataset_name,
        save_dir,
        doc_column=None,
        label_column=None,
        encoding="cp1252",
        **preprocessing_args,
    ):
        # Check if data is a DataFrame
        if isinstance(data, pd.DataFrame):
            if doc_column is None:
                raise ValueError("doc_column n must be specified for DataFrame input")
            documents = [
                self.clean_text(str(row[doc_column])) for _, row in data.iterrows()
            ]
            if label_column is None:
                print(
                    "You have not specified any labels. The dataset will be created without labels"
                )
            labels = (
                data[label_column].tolist()
                if label_column
                else np.repeat(None, len(documents)).to_list()
            )
        elif isinstance(data, list):  # Assuming data is a list of documents
            documents = [self.clean_text(doc) for doc in data]
            labels = np.repeat(None, len(documents)).to_list()
        else:
            raise TypeError("data must be a pandas DataFrame or a list of documents")

        # Save documents and labels to files
        documents_path = f"{save_dir}/{dataset_name}_corpus.txt"
        labels_path = f"{save_dir}/{dataset_name}_labels.txt"

        with open(documents_path, "w", encoding=encoding, errors="replace") as file:
            for doc in documents:
                file.write(doc + "\n")

        with open(labels_path, "w", encoding=encoding, errors="replace") as file:
            for label in labels:
                file.write(str(label) + "\n")

        # Preprocess the dataset
        preprocessor = Preprocessing(**preprocessing_args)

        dataset = preprocessor.preprocess_dataset(
            documents_path=documents_path, labels_path=labels_path
        )

        # Save the preprocessed dataset
        dataset.save(f"{save_dir}/{dataset_name}")

        return dataset


# Usage
if __name__ == "__main__":
    dataset = TMDataset()
    dataset.fetch_dataset("Spotify")
