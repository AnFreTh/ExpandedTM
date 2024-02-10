import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ExpandedTM.models.WordCluTM import WordCluTM
from ExpandedTM.data_utils.dataset import (
    TMDataset,
)
import string
import random
import pandas as pd  # Adjust the import according to your package structure


class TestWordCluTM(unittest.TestCase):

    def setUp(self):
        # Initialize your WordCluTM model with default or specific parameters
        self.model = WordCluTM(
            num_topics=10, vector_size=50, window=5, min_count=1, workers=2
        )

        # Create a mock dataset, assuming your dataset.get_corpus() returns a list of tokenized sentences
        self.mock_dataset = MagicMock(spec=TMDataset)
        # Generate random words and create a dataframe
        random_texts = [
            "sample text " + "".join(random.choices(string.ascii_lowercase, k=250))
            for _ in range(50)
        ]
        self.mock_dataset.dataframe = pd.DataFrame({"text": random_texts})
        # Generate a mock vocabulary
        self.mock_dataset.get_vocabulary.return_value = list(
            set([word for word in random_texts])
        )
        self.mock_dataset.get_corpus.return_value = [
            ["word1", "word2", "word3"],
            ["word4", "word5"],
        ]

    @patch("umap.umap_.UMAP")  # Adjust the import path
    def test_dim_reduction(self, mock_umap):
        # Mock UMAP fit_transform method to return a predetermined value
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.random.rand(
            10, 7
        )  # Assuming 10 words, reduced to 7 dimensions
        mock_umap.return_value = mock_reducer

        # Call _dim_reduction with a dummy embeddings array
        dummy_embeddings = np.random.rand(
            10, 50
        )  # Assuming 10 words, 50-dimensional embeddings
        reduced_embeddings = self.model._dim_reduction(dummy_embeddings)

        # Assertions
        mock_umap.assert_called_once_with(**self.model.umap_args)
        mock_reducer.fit_transform.assert_called_once_with(dummy_embeddings)
        self.assertEqual(
            reduced_embeddings.shape, (10, 7)
        )  # Check if the dimensionality reduction output shape is correct

    @patch("sklearn.mixture.GaussianMixture")  # Adjust the import path
    def test_clustering(self, mock_gmm):
        # Mock GaussianMixture and its methods
        mock_gmm_instance = MagicMock()
        mock_gmm_instance.predict_proba.return_value = np.random.rand(
            10, self.model.n_topics
        )  # Soft labels
        mock_gmm_instance.predict.return_value = np.random.randint(
            0, self.model.n_topics, 10
        )  # Hard labels
        mock_gmm.return_value = mock_gmm_instance

        # Assuming reduced_embeddings is a required attribute for _clustering
        self.model.reduced_embeddings = np.random.rand(
            10, 7
        )  # Dummy reduced embeddings

        soft_labels, labels = self.model._clustering()

        # Assertions
        mock_gmm.assert_called_once_with(**self.model.gmm_args)
        mock_gmm_instance.fit.assert_called_once_with(self.model.reduced_embeddings)
        self.assertEqual(
            soft_labels.shape, (10, self.model.n_topics)
        )  # Check soft labels shape
        self.assertEqual(labels.shape, (10,))  # Check hard labels shape


if __name__ == "__main__":
    unittest.main()
