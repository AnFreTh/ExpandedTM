# ExpandedTM
Python library for topic modeling with extended corpora.
The implemented models include CEDC, DCTE, KMeans, Top2Vec.


Available Models
=================

| **Name**                                                                                                                                      | **Implementation**                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [CEDC](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)     | Topics in the Haystack                                                  |
| [DCTE](https://arxiv.org/pdf/2212.09422.pdf)                                                                                                  | Human in the Loop                                                       |
| [KMeansTM](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1) | Simple Kmeans followed by c-tfidf                                       |
| SomTM                                                                                                                                         | Self organizing map followed by c-tfidf                                 |
| [CBC](https://ieeexplore.ieee.org/abstract/document/10066754)                                                                                 | Coherence based document clustering                                     |
| TNTM_bow                                                                                                                                      | Transformer-Representation Neural Topic Model using bag-of-words        |
| TNTM_SentenceTransformer                                                                                                                      | Transformer-Representation Neural Topic Model using SentenceTransformer |

Available (Additional) Metrics
=================

| **Name**                                                                                                                                                 | **Description**                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ISIM](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)                | Average cosine similarity of top words of a topic to an intruder word.                                                                                                 |
| [INT](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)                 | For a given topic and a given intruder word, Intruder Accuracy is the fraction of top words to which the intruder has the least similar embedding among all top words. |
| [ISH](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)                 | calculates the shift in the centroid of a topic when an intruder word is replaced.                                                                                     |
| [COH](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1)                 | Embedding Coherence                                                                                                                                                    |
| [Embedding Coherence](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00506/118990/Topics-in-the-Haystack-Enhancing-Topic-Quality?searchresult=1) | Cosine similarity between the centroid of the embeddings of the stopwords and the centroid of the topic.                                                               |
| NPMI                                                                                                                                                     | Classical NPMi coherence computed on the scource corpus.                                                                                                               |




Available Datasets
=================

| **Name**                  | **Description**                                         |
| ------------------------- | ------------------------------------------------------- |
| Spotify                   | Random subset of Spotify song lyrics.                   |
| Reuters                   | Standard Reuters dataset preprocessed for octis format. |
| Spotify_most_popular      | Spotify lyrics of most popular songs.                   |
| Poliblogs                 | Standard Poliblogs dataset.                             |
| Reddit_GME                | Filtered sample for "Gamestop" (GME) from the Subreddit "r/wallstreetbets". Sample is taken from the thread "What are your moves tomorrow?". It is covering the time around the GME short squeeze of 2021 |
| Stocktwits_GME            | Filtered sample for "Gamestop" (GME) from Stocktwits. It is covering the time around the GME short squeeze of 2021 |

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
