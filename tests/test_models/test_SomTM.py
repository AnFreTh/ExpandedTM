import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from ExpandedTM.models.som import SOMTM
from ExpandedTM.data_utils.dataset import TMDataset


class TestSOMTM(unittest.TestCase):
    def setUp(self):
        # Mock the TMDataset with initial embeddings of shape (25, 128)
        self.mock_dataset = MagicMock(spec=TMDataset)
        self.mock_dataset.get_embeddings.return_value = np.random.rand(25, 128)
        self.mock_dataset.dataframe = pd.DataFrame({"text": ["sample text"] * 25})

        # Initialize the KmeansTM model
        self.model = SOMTM(m=20, n=1, dim=384, n_iterations=3)

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
