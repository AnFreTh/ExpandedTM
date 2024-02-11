import unittest
from unittest.mock import patch, MagicMock
import string
import random
from ExpandedTM.models.som import SOMTM
from ExpandedTM.data_utils.dataset import TMDataset
import pandas as pd
import numpy as np


class TestSOMTM(unittest.TestCase):
    def setUp(self):
        # Mock the TMDataset with initial embeddings of shape (25, 128)
        self.n_topics = 10
        self.n_words_per_topic = 23
        self.n_documents = 500

        self.mock_dataset = MagicMock(spec=TMDataset)
        # Prepare diverse labels and text data
        labels = ["A", "B", "C", "D", "E"]
        text_data = [
            " ".join(
                "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 15)))
                for _ in range(random.randint(5, 10))  # Each document has 5-10 words
            )
            for _ in range(50)  # 50 documents
        ]

        label_data = [labels[i % len(labels)] for i in range(50)]
        self.mock_dataset.dataframe = pd.DataFrame(
            {"text": text_data, "label_text": label_data}
        )

        # Set vocabulary and corpus
        self.mock_dataset.get_vocabulary = lambda: list(
            set(word for text in text_data for word in text.split())
        )
        self.mock_dataset.get_corpus = lambda: [text.split() for text in text_data]

        self.mock_dataset.get_embeddings.return_value = np.random.rand(
            self.n_documents, 128
        )

        # Initialize the KmeansTM model
        self.model = SOMTM(m=5, n=2, dim=384, n_iterations=10)

    def test_prepare_data(self):
        # Test data preparation
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.assertIsNotNone(self.model.embeddings)
        self.assertIsNotNone(self.model.dataframe)

    @patch("umap.umap_.UMAP")
    def test_dim_reduction(self, mock_umap):
        # Test dimensionality reduction
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.model._dim_reduction()
        mock_umap.assert_called_once()
        self.assertIsNotNone(self.model.reduced_embeddings)

    def test_train_model(self):
        output = self.model.train_model(self.mock_dataset)
        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)


if __name__ == "__main__":
    unittest.main()
