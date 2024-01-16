## Available Datasets
-------------------

Xou can leverage and load all dataset available in [OCTIS](https://aclanthology.org/2021.eacl-demos.31.pdf).

Additionally there are some preprocessed datasets available in ExpandedTM:

| **Name**            | **#Docs** | **# Words** | **# Labels** |
| ------------------- | --------- | ----------- | ------------ |
| Reuters             | 8929      | 24803       | 60           |
| Poliblogs           | 13246     | 70726       | 2            |
| Spotify             | 4323      | 12621       | cont.        |
| Spofiy_most_popular | 5065      | 63211       | cont.        |



Load a preprocessed dataset
----------------------------

To load one of the already preprocessed datasets as follows:

    ```python
   from ExpandedTM.data_utils import TMDataset
   dataset = TMDataset()
   dataset.fetch_dataset("Spotify")
   ```

Just use one of the dataset names listed above. Note: it is case-sensitive!


Load a custom preprocessed dataset
----------------------------

Otherwise, you can load a custom preprocessed dataset in the following way, by simply using a pandas dataframe:

    ```python
   from ExpandedTM.data_utils import TMDataset
   dataset = TMDataset()
   dataset = dataset.create_load_save_dataset(my_data, "test",
        "..",
        doc_column="Documents",
        label_column="Labels",
        )
    ```

