# ExpandedTM
Python library for topic modeling with extended corpora
The implemented models include CEDC, DCTE, KMeans, Top2Vec.


Available Models
=================

| **Name**                              | **Implementation**                 |
| ------------------------------------- | ---------------------------------- |
| CEDC                                  | Topics in the Haystack             |
| DCTE                                  | Human in the Loop                  |
| KMeans                                | Simple Kmeans followed by c-tfidf  |




Available (Additional) Metrics
=================

| **Name**     | **Description**                                                                                                                                                        |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ISIM         | Average cosine similarity of top words of a topic to an intruder word.                                                                                                 |
| INT          | For a given topic and a given intruder word, Intruder Accuracy is the fraction of top words to which the intruder has the least similar embedding among all top words. |
| COH          | Embedding Coherence                                                                                                                                                    |
| Expressivity | Cosine similarity between the centroid of the embeddings of the stopwords and the centroid of the topic.                                                               |



## Usage

To use these models, follow the steps below:

1. Import the necessary modules:

    ```python
    from octis.models.CEDC import CEDC
    from octis.models.KmeansTM import KmeansTM
    from octis.models.DCTE import DCTE
    from octis.dataset.dataset import Dataset
    ```

2. Get your dataset and data directory:

    ```python
    data_dir = './preprocessed_datasets'

    data = Dataset()

    data.fetch_dataset("20NewsGroup")
    ```

3. Choose the model you want to use and train it:

    ```python
    model = KmeansTM(num_topics=20)
    output = model.train_model(dataset)
    ```

4. Evaluate the model using either Octis evaluation metrics or newly defined ones such as INT or ISIM:

    ```python
    from octis.evaluation_metrics.intruder_metrics import ISIM, INT

    metric = ISIM(dataset)
    metric.score(output)
    ```


