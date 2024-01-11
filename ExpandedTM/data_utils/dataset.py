from octis.dataset.dataset import Dataset as OCTISDataset
import os
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from octis.preprocessing.preprocessing import Preprocessing


import os
import pickle
import pandas as pd


class TMDataset(OCTISDataset):
    def __init__(self):
        super().__init__()

    def fetch_dataset(self, name):
        # Check if the path is relative to the package datasets
        self.name = name
        package_dataset_path = self.get_package_dataset_path(name)
        print(package_dataset_path)

        if os.path.exists(package_dataset_path):
            # If the dataset exists in the package, load it from there
            super().load_custom_dataset_from_folder(package_dataset_path)
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
    def get_package_dataset_path(relative_path):
        # Define the base path to your package's dataset directory
        base_path = os.path.join(os.path.dirname(__file__), "preprocessed_datasets")

        # Combine the base path with the relative path
        return os.path.join(base_path, relative_path)

    def get_embeddings(self, embedding_model_name):
        # Construct the dataset folder path
        dataset_folder = self.get_package_dataset_path(self.name)

        # Ensure the dataset folder exists or create it if it doesn't
        os.makedirs(dataset_folder, exist_ok=True)

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
            labels = data[label_column].tolist() if label_column else None
        elif isinstance(data, list):  # Assuming data is a list of documents
            documents = [self.clean_text(doc) for doc in data]
            labels = None
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
