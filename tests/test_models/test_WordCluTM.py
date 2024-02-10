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

    def test_train_word2vec(self):
        # Train Word2Vec and check if embeddings are generated
        self.model.train_word2vec(self.mock_dataset.get_corpus())
        self.assertIsNotNone(self.model.word2vec_model)
        self.assertIn("word1", self.model.word2vec_model.wv.key_to_index)

    def test_dim_reduction_and_clustering(self):
        # Assuming train_word2vec and prepare_data have run successfully
        self.model.train_word2vec(self.mock_dataset.get_corpus())
        word_embeddings = np.array(
            [
                self.model.word2vec_model.wv[word]
                for word in self.model.word2vec_model.wv.index_to_key
            ]
        )

        # Perform dimensionality reduction
        reduced_embeddings = self.model._dim_reduction(word_embeddings)
        self.assertEqual(
            reduced_embeddings.shape,
            (len(word_embeddings), self.model.umap_args["n_components"]),
        )

        # Perform clustering on the reduced embeddings
        self.model.reduced_embeddings = reduced_embeddings
        soft_labels, labels = self.model._clustering()
        self.assertEqual(len(labels), len(word_embeddings))
        self.assertEqual(soft_labels.shape, (len(word_embeddings), self.model.n_topics))

    def test_train_model(self):
        # Train the model on the dataset and check if output is generated as expected
        output = self.model.train_model(self.mock_dataset)
        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)
        self.assertIn("topic-document-matrix", output)
        self.assertTrue(self.model.trained)


if __name__ == "__main__":
    unittest.main()
