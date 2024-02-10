import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ExpandedTM.models.WordCluTM import WordCluTM
from ExpandedTM.data_utils.dataset import TMDataset


class TestWordCluTM(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        self.mock_dataset = MagicMock(spec=TMDataset)
        # Generate mock corpus
        self.mock_dataset.get_corpus.return_value = [
            ["word1", "word2", "word3"],
            ["word4", "word5", "word6"],
            # Add more mock sentences as needed
        ]
        # Generate a mock dataframe
        self.mock_dataset.get_dataframe.return_value = MagicMock()
        self.mock_dataset.get_dataframe.return_value.dataframe = MagicMock()

        # Initialize the WordCluTM model
        self.model = WordCluTM(num_topics=10)

    def test_train_word2vec(self):
        # Test Word2Vec training
        sentences = self.mock_dataset.get_corpus()
        self.model.train_word2vec(sentences)
        self.assertIsNotNone(self.model.word2vec_model)

    def test_train_model(self, mock_clustering, mock_dim_reduction):
        # Mock _dim_reduction to return reduced embeddings
        mock_dim_reduction.return_value = np.random.rand(3, 3)
        # Mock _clustering to return labels
        mock_clustering.return_value = (np.random.rand(3, 3), np.array([0, 1, 2]))

        # Run the training process
        output = self.model.train_model(self.mock_dataset)

        # Add assertions here to validate the output of the training process
        # For example:
        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)
        self.assertIn("topic_dict", output)
        self.assertIn("topic-document-matrix", output)


if __name__ == "__main__":
    unittest.main()
