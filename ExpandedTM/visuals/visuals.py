import matplotlib.pyplot as plt
from wordcloud import WordCloud
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


def visualize_topic_model(
    model, three_dim=False, reduce_first=False, reducer="umap", port=8050
):
    """_summary_

    Args:
        model (AbstractModel): trained topic model
        three_dim (bool, optional): whether to cisualize 3-dimensional or 2-dimensional. Defaults to False.
        reduce_first (bool, optional): Whether the document embeddings are first dimensionality reduced and then the topical centroids are computed. Defaults to False.
        reducer (str, optional): Which model is used to perform dimensionality reduction. Defaults to "umap".
        port (int, optional): port of dash plot. Defaults to 8050.
    """
    assert (
        model.trained
    ), "Be sure to only pass a trained model to the visualization function"

    if three_dim:
        _visualize_topic_model_3d(model, reduce_first, reducer, port)
    else:
        _visualize_topic_model_2d(model, reduce_first, reducer, port)


def visualize_topics(model, three_dim=False, reducer="umap", port=8050):
    """_summary_

    Args:
        model (AbstractModel): trained topic model
        three_dim (bool, optional): whether to cisualize 3-dimensional or 2-dimensional. Defaults to False.
        reducer (str, optional): Which model is used to perform dimensionality reduction. Defaults to "umap".
        port (int, optional): port of dash plot. Defaults to 8050.
    """

    assert (
        model.trained
    ), "Be sure to only pass a trained model to the visualization function"

    if three_dim:
        _visualize_topics_3d(model, reducer, port)
    else:
        _visualize_topics_2d(model, reducer, port)
