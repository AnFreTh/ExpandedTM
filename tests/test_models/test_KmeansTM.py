import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from ExpandedTM.models.Kmeans import KmeansTM
from ExpandedTM.data_utils.dataset import TMDataset


class TestKmeansTM(unittest.TestCase):
    def setUp(self):
        # Mock the TMDataset with initial embeddings of shape (25, 128)
        self.mock_dataset = MagicMock(spec=TMDataset)
        self.mock_dataset.get_embeddings.return_value = np.random.rand(25, 128)
        self.mock_dataset.dataframe = pd.DataFrame({"text": ["sample text"] * 25})

        # Initialize the KmeansTM model
        self.model = KmeansTM(num_topics=10)

    def test_prepare_data(self):
        # Test data preparation
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.assertIsNotNone(self.model.embeddings)
        self.assertIsNotNone(self.model.dataframe)

    @patch("ExpandedTM.models.CEDC.umap.UMAP")
    def test_dim_reduction(self, mock_umap):
        # Test dimensionality reduction
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.model._dim_reduction()
        mock_umap.assert_called_once()
        self.assertIsNotNone(self.model.reduced_embeddings)

    @patch("ExpandedTM.models.Kmeans.KMeans")
    def test_clustering(self, mock_kmeans):
        # Test clustering
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.model._dim_reduction()
        self.model._clustering()
        mock_kmeans.assert_called_once()
        self.assertIsNotNone(self.model.labels)

    @patch("umap.umap_.UMAP.fit_transform")
    def test_train_model(self, mock_umap_fit_transform):
        mock_umap_fit_transform.return_value = np.random.rand(25, 15)

        output = self.model.train_model(self.mock_dataset)

        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)


if __name__ == "__main__":
    unittest.main()
