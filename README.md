# ExpandedTM
Python library for topic modeling with extended corpora.
The implemented models include CEDC, DCTE, KMeans, Top2Vec.


Available Models
=================

| **Name** | **Implementation**                |
| -------- | --------------------------------- |
| CEDC     | Topics in the Haystack            |
| DCTE     | Human in the Loop                 |
| KMeansTM | Simple Kmeans followed by c-tfidf |




Available (Additional) Metrics
=================

| **Name**            | **Description**                                                                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ISIM                | Average cosine similarity of top words of a topic to an intruder word.                                                                                                 |
| INT                 | For a given topic and a given intruder word, Intruder Accuracy is the fraction of top words to which the intruder has the least similar embedding among all top words. |
| COH                 | Embedding Coherence                                                                                                                                                    |
| Embedding Coherence | Cosine similarity between the centroid of the embeddings of the stopwords and the centroid of the topic.                                                               |
| NPMI                | Classical NPMi coherence computed on the scource corpus.                                                                                                               |




Available Datasets
=================

| **Name**             | **Description**                                         |
| -------------------- | ------------------------------------------------------- |
| Spotify              | Random subset of Spotify song lyrics.                   |
| Reuters              | Standard Reuters dataset preprocessed for octis format. |
| Spotify_most_popular | Spotify lyrics of most popular songs.                   |
| Poliblogs            | Standard Poliblogs dataset.                             |

## Usage

To use these models, follow the steps below:

1. Import the necessary modules:

    ```python
    from ExpandedTM.models import CEDC, KmeansTM, DCTE
    from ExpandedTM.data_utils import TMDataset
    ```

2. Get your dataset and data directory:

    ```python
    data = TMDataset()

    data.fetch_dataset("20NewsGroup")
    ```

3. Choose the model you want to use and train it:

    ```python
    model = CEDC(num_topics=20)
    output = model.train_model(dataset)
    ```

4. Evaluate the model using either Octis evaluation metrics or newly defined ones such as INT or ISIM:

    ```python
    from ExpandedTM.metrics import ISIM, INT

    metric = ISIM(dataset)
    metric.score(output)
    ```

5. Score per topic


    ```python
    metric.score_per_topic(output)
    ```

6. Visualize the results:
7. 
    ```python
    from ExpandedTM.visuals import visualize_topic_model, visualize_topics

    visualize_topic_model(
        model, 
        reduce_first=True, 
        port=8051,
        )
    ```
