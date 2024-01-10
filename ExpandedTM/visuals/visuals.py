import umap.umap_ as umap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from ._interactive import (
    _visualize_topic_model_2d,
    _visualize_topic_model_3d,
    _visualize_topics_2d,
    _visualize_topics_3d,
)


def visualize_topics_as_wordclouds(model, max_words=100):
    """
    Generates word clouds for each topic in the model.

    Parameters:
        model (KmeansTM): A trained instance of the KmeansTM model.
        max_words (int): Maximum number of words to include in the word cloud.
    """
    assert (
        hasattr(model, "output") and "topic_dict" in model.output
    ), "Model must have been trained with topics extracted."

    topics = model.output["topic_dict"]

    for topic_id, topic_words in topics.items():
        # Generate a word frequency dictionary for the topic
        word_freq = {word: weight for word, weight in topic_words}

        # Create and display the word cloud
        wordcloud = WordCloud(
            width=800, height=400, max_words=max_words
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(f"Topic {topic_id}")
        plt.axis("off")
        plt.show()


def visualize_clusters_2d(model):
    """
    Visualizes the clusters created by the model in a 2D space using UMAP for dimensionality reduction.

    Parameters:
        model (KmeansTM): A trained instance of the KmeansTM model.
    """
    assert hasattr(model, "embeddings"), "Model must have 'embeddings' attribute."
    assert hasattr(model, "labels"), "Model must have 'labels' attribute."

    # Reduce dimensionality to 2D
    umap_model = umap.UMAP(n_components=2, random_state=42)
    reduced_embeddings = umap_model.fit_transform(model.embeddings)

    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=model.labels,
        cmap="viridis",
        s=5,
    )
    plt.title("Cluster Visualization in 2D Space")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(scatter, label="Cluster")
    plt.show()


def visualize_topic_centroids(model):
    """
    Visualizes a Topic Distance Map using UMAP.

    Parameters:
        model (KmeansTM): A trained instance of the KmeansTM model.
    """
    # Check if centroids are available
    if hasattr(model, "topic_centroids"):
        centroids = model.topic_centroids
    else:
        # Ensure embeddings and labels are present
        assert hasattr(model, "embeddings") and hasattr(
            model, "labels"
        ), "Model must have 'embeddings' and 'labels' attributes."

        # Compute centroids from embeddings and labels
        unique_labels = np.unique(model.labels)
        centroids = np.array(
            [
                model.embeddings[model.labels == label].mean(axis=0)
                for label in unique_labels
            ]
        )

    # Use UMAP for dimensionality reduction
    umap_model = umap.UMAP(n_components=2, random_state=42)
    reduced_centroids = umap_model.fit_transform(centroids)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], marker="o")
    for i, centroid in enumerate(reduced_centroids):
        plt.text(
            centroid[0], centroid[1], str(i), fontdict={"weight": "bold", "size": 9}
        )

    plt.title("Topic Distance Map with UMAP")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()


def visualize_topic_model(model, three_dim=True, reduce_first=False, port=8050):
    if three_dim:
        _visualize_topic_model_3d(model, reduce_first, port)
    else:
        _visualize_topic_model_2d(model, reduce_first, port)


def visualize_topics(model, three_dim=True, port=8050):
    if three_dim:
        _visualize_topics_3d(model, port)
    else:
        _visualize_topics_2d(model, port)
