from multiprocessing.process import parent_process
import torch
from torch import nn
import numpy as np
from collections import namedtuple
import numpy as np
import ExpandedTM.models.TNTM.TNTM_inference as TNTM_inference
import ExpandedTM.models.TNTM.Initialization as init


class TNTM_SentenceTransformer:
    def __init__(
        self,
        embedding_model_name_vocabulary: str = "all-MiniLM-L6-v2",
        embedding_model_name_documents: str = "all-MiniLM-L6-v2",
        num_topics: int = 20,
        save_path: str = None,
        n_dims: int = 11,
        n_hidden_units: int = 200,
        n_encoder_layers: int = 3,
        enc_lr: float = 1e-3,
        dec_lr: float = 1e-3,
        n_epochs=100,
        batch_size: int = 128,
        dropout_rate_encoder: float = 0.3,
        prior_variance=0.995,
        prior_mean=None,
        n_topwords=10,
        device=None,
        validation_set_size: float = 0.2,
        early_stopping: bool = True,
        n_epochs_early_stopping: int = 10,
        umap_hyperparams: dict = {"n_neighbors": 15, "min_dist": 0.1},
    ):
        """
        :param str embedding_model_name_vocabulary: Name of the embedding model to use for the vocabulary
        :param str embedding_model_name_documents: Name of the embedding model to use for the documents
        :param int n_topics: the number of topics to be estimated
        :param str save_path: Path to save the model
        :param int n_dims: the number of dimensions of the word embedding space to operate in
        :param int n_hidden_units: number of hidden units in the encoder
        :param int n_encoder_layers: number of layers in the encoder
        :param float enc_lr: learning rate for the encoder. Important to tune on new data.
        :param float dec_lr: learning rate to optimize the paramters of the Gaussians representing the topics in the embedding space. Important to tune on new data.
        :param int: n_epochs: number of epochs to train
        :param int batch_size: Batch size to train encoder and decoder. Tune wrt. learning rate.
        :param float dropout_rate_encoder: Dropout rate in the encoder
        :param prior_variance: variance of the logistic normal prior on the document-topic distributions
        :type prior_variance: if float, assume a symmetric prior with the given variance. Else given by a tensor of shape (1, n_topics)
        :param prior_mean: mean of the logistic normal prior on the document-topic distribution. Use zero per default. Otherwise, this parameter is given by a tensor of shape (1, n_topics)
        :param int n_topwords: number of topwords to extract per topic
        :param device: "cpu" or "cuda". If "none", use gpu if available, else cpu
        :param float: validation_set_size: Fraction of the used dataset for validation
        :param bool early_stopping : Whether early stopping based on the median validation loss should be done
        :param int n_epochs_early_stopping: Patience paramter for early stopping, i.e. for how many epochs to wait until the next decrease in median validation loss has to happen
        :param dict umap_hyperparams: Hyperparameters for the UMAP algorithm used for visualization. See https://umap-learn.readthedocs.io/en/latest/parameters.html for more details
        """

        self.embedding_model_name_vocabulary = embedding_model_name_vocabulary
        self.embedding_model_name_documents = embedding_model_name_documents

        self.n_topics = num_topics
        self.save_path = save_path
        self.n_dims = n_dims
        self.n_hidden_units = n_hidden_units
        self.n_encoder_layers = n_encoder_layers
        self.enc_lr = enc_lr
        self.dec_lr = dec_lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_topwords = n_topwords
        self.dropout_rate_encoder = dropout_rate_encoder
        self.validation_set_size = validation_set_size
        self.early_stopping = early_stopping
        self.n_epochs_early_stopping = n_epochs_early_stopping
        self.umap_hyperparams = umap_hyperparams
        self.trained = False

        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        assert type(prior_variance) in [
            float,
            torch.Tensor,
        ], "prior variance must be a float or a tensor of of shape (1, n_topics)"

        if type(prior_variance) == float:
            self.prior_var = (
                torch.Tensor(1, self.n_topics).fill_(prior_variance).to(device)
            )  # initialize tensor with prior variance

        else:
            self.prior_var = prior_variance.to(device)
        self.prior_logvar = torch.log(self.prior_var).to(device)

        if prior_mean == None:
            self.prior_mean = torch.Tensor(1, self.n_topics).fill_(0)

        else:
            self.prior_mean = prior_mean.to(device)

    def train_model(self, dataset, n_words=20):
        """
        :param dataset : dataset object
        :param int n_words : number of top words to extract per topic
        """

        vocab = dataset.get_vocabulary()
        corpus = dataset.get_corpus()
        embeddings_vocab = dataset.get_embeddings_vocabulary(
            self.embedding_model_name_vocabulary
        )
        embeddings_docs = dataset.get_embeddings(self.embedding_model_name_documents)

        if n_words is not None:
            self.n_topwords = n_words

        if type(embeddings_vocab) == np.ndarray:
            embeddings_vocab = torch.tensor(embeddings_vocab).to(self.device)

        if type(embeddings_docs) == np.ndarray:
            embeddings_docs = torch.tensor(embeddings_docs).to(self.device)

        self.vocab = vocab
        self.corpus = corpus
        self.embeddings_vocab = embeddings_vocab
        self.embeddings_docs = embeddings_docs

        self.vocab = vocab
        self.corpus = corpus
        self.embeddings = embeddings_vocab
        self.document_embeddings = embeddings_docs

        self.word2idx = {
            word: i for i, word in enumerate(self.vocab)
        }  # map index of word in the vocabulary to the actual word
        self.idx2word = {
            i: word for i, word in enumerate(self.vocab)
        }  # map word to its index in the vocabulary

        self.embedding_ten = self.embeddings.to(self.device)

        # represent each document by a row of length len(vocab) where the entry b in the i-th position indicates that the i-th word occurs b times in the document of row k
        bow_ten = torch.zeros(len(corpus), len(vocab))
        corpus_idx = [[self.word2idx[word] for word in doc] for doc in self.corpus]
        for i, doc in enumerate(corpus_idx):
            for word in doc:
                bow_ten[i, word] += 1

        self.bow_ten = bow_ten.to_sparse()

        # compute the low-dimensional embeddings and the initial topic assignments to use later
        init_in = init.Initializer(
            self.embedding_ten.cpu().detach().numpy(),
            n_topics=self.n_topics,
            n_dims=self.n_dims,
        )
        (
            embeddings_proj,
            mus_init,
            L_lower_init,
            log_diag_init,
            bic,
        ) = init_in.reduce_dim_and_cluster(umap_hyperparams=self.umap_hyperparams)

        embeddings_proj_ten = torch.tensor(embeddings_proj).to(self.device)

        # initialize training of the VAEa

        train_ds, val_ds, test_ds = TNTM_inference.train_test_split(
            list(zip(self.document_embeddings, self.bow_ten)),
            1 - self.validation_set_size,
            self.validation_set_size,
            self.batch_size,
        )

        train_config = {
            "num_input": len(self.vocab),
            "n_hidden_block": self.n_hidden_units,
            "n_skip_layers": self.n_encoder_layers,
            "n_topics": self.n_topics,
            "drop_rate_en": self.dropout_rate_encoder,
            "init_mult": 1,
            "vocab_size": len(self.vocab),
            "embedding_dim": self.n_dims,
            "sentence_transformer_hidden_dim": embeddings_docs.shape[1],
            "early_stopping": self.early_stopping,
            "n_epochs_early_stopping": self.n_epochs_early_stopping,
        }

        self.train_config = namedtuple("train_config", train_config.keys())(
            *train_config.values()
        )

        self.model = TNTM_inference.TNTM_sentence_transformer_precomputed(
            embeddings=embeddings_proj_ten.to(self.device),
            mus_init=mus_init.to(self.device),
            lower_init=L_lower_init.to(self.device),
            log_diag_init=log_diag_init,
            config=self.train_config,
            prior_mean=self.prior_mean.to(self.device),
            prior_variance=self.prior_var.to(self.device),
        ).to(self.device)

        opt1 = torch.optim.Adam(
            self.model.encoder.parameters(), lr=self.enc_lr, betas=(0.99, 0.999)
        )
        opt2 = torch.optim.Adam(self.model.decoder.parameters(), lr=self.dec_lr)

        # training

        TNTM_inference.train_loop(
            model=self.model,
            optimizer1=opt1,
            optimizer2=opt2,
            trainset=train_ds,
            valset=val_ds,
            print_mod=1,
            device=self.device,
            n_epochs=self.n_epochs,
            save_path=self.save_path,
            config=self.train_config,
            sparse_ten=True,
        )

        # get topic paramters

        self.mus_res = self.model.decoder.mus.detach()
        self.L_lower_res = self.model.decoder.L_lower.detach()
        self.log_diag_res = self.model.decoder.log_diag.detach()

        probs = TNTM_inference.get_topic_word_mat(  # get the probability of each word under each topic, has shape (n_topics, len(vocab))
            mus_res=self.mus_res,
            L_lower_res=self.L_lower_res,
            D_log_res=self.log_diag_res,
            emb_vocab_mat=embeddings_proj_ten,
            config=self.train_config,
        )

        topwords, probs_new = [], []

        for i in range(self.n_topics):
            topwords.append(
                [
                    self.idx2word[idx.item()]
                    for idx in np.argsort(probs[i])[-self.n_topwords :]
                ]
            )
            probs_new.append(np.sort(probs[i])[-self.n_topwords :])

        topic_dict = {}

        for i in range(self.n_topics):
            topic_dict[i] = list(zip(topwords[i], probs_new[i]))

        self.output = {}
        self.output["topics"] = topwords
        self.output["topic-word-matrix"] = probs
        self.output["topic_dict"] = topic_dict

        self.trained = True

        self.labels = probs.argmax(dim=0)

        return self.output
