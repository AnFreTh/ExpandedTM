import torch
from typing import List
import matplotlib.pyplot as plt
from typing import Tuple
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd


class FeatureNN(torch.nn.Module):
    """
    A feature neural network module using PyTorch.

    This class defines a neural network with customizable layers, activations,
    and dropout for feature extraction and transformation.

    Attributes:
        layers (torch.nn.ModuleList): A list of layers in the neural network.
        output_layer (torch.nn.Linear): The output layer of the neural network.
        dropout (torch.nn.Dropout): Dropout layer to prevent overfitting.
        activations (List[torch.nn.Module]): List of activation functions for each layer.

    Args:
        shallow_units (int): Number of units in the initial layer.
        output_size (int, optional): Size of the output layer. Defaults to 1.
        input_size (int, optional): Size of the input layer. Defaults to 1.
        hidden_units (Tuple, optional): Tuple specifying the number of units in each hidden layer.
        activations (List[torch.nn.Module], optional): List of activation functions to use for each layer. Defaults to [torch.nn.ReLU()].
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.3.
    """

    def __init__(
        self,
        shallow_units: int,
        output_size: int = 1,
        input_size: int = 1,
        hidden_units: Tuple = (),
        activations: List = [torch.nn.ReLU()],
        dropout: float = 0.3,
    ):
        super().__init__()

        # Define Layers
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    shallow_units if i == 0 else hidden_units[i - 1], hidden_units[i]
                )
                for i in range(len(hidden_units))
            ]
        )

        self.layers.insert(0, torch.nn.Linear(input_size, shallow_units))
        self.output_layer = torch.nn.Linear(hidden_units[-1], output_size)

        # Dropout and activation
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activations = activations

    def forward(self, x):
        x = x.unsqueeze(1)
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x))
            x = self.dropout(x)
        return self.output_layer(x)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        with torch.no_grad():
            x_axis = torch.linspace(0, 1, 500).reshape(-1, 1)

            ax.plot(
                x_axis,
                self.forward(x_axis).squeeze().reshape(-1, 1),
                linestyle="solid",
                linewidth=1,
                color="red",
            )

    def plot_data(self, ax=None, x=None, y=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        with torch.no_grad():
            x_axis = torch.linspace(-1, 1, 500).reshape(-1, 1)

            ax.plot(
                x_axis,
                self.forward(x_axis).squeeze().reshape(-1, 1),
                linestyle="solid",
                linewidth=1,
                color="red",
            )

            ax.scatter(x, y, color="gray", s=2, alpha=0.3)


class NeuralAdditiveModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        shallow_units: int or List[int],  # Added shallow_units to the parameters
        hidden_units: List[int] = None,
        feature_dropout: float = 0.0,
        hidden_dropout: float = 0.3,
        activations: List[torch.nn.Module] = [torch.nn.ReLU()],
        out_activation=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.out_activation = out_activation

        if isinstance(shallow_units, list):
            assert (
                len(shallow_units) == input_size
            ), "shallow_units list must match input_size"
        elif isinstance(shallow_units, int):
            shallow_units = [shallow_units for _ in range(input_size)]

        hidden_units = hidden_units or [128, 64]

        self.feature_nns = torch.nn.ModuleList(
            [
                self._build_feature_nn(
                    shallow_units[i],
                    hidden_units,
                    output_size,
                    activations,
                    hidden_dropout,
                )
                for i in range(input_size)
            ]
        )
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(output_size))

    def _build_feature_nn(
        self, shallow_unit, hidden_units, output_size, activations, dropout
    ):
        layers = [torch.nn.Linear(1, shallow_unit), activations[0]]
        for i in range(len(hidden_units) - 1):
            layers.append(torch.nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(
                activations[i + 1] if i + 1 < len(activations) else activations[-1]
            )
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_units[-1], output_size))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        feature_outputs = torch.cat(
            [nn(x[:, i : i + 1]) for i, nn in enumerate(self.feature_nns)], dim=1
        )
        output = feature_outputs.sum(dim=1, keepdim=True) + self.bias
        if self.out_activation is not None:
            output = self.out_activation(output)
        return output

    def plot(self):
        self.eval()
        with torch.no_grad():
            if len(self.feature_nns) > 1:
                fig, axes = plt.subplots(len(self.feature_nns), 1, figsize=(10, 7))
                for i, ax in enumerate(axes.flat):
                    component = self.feature_nns[i]
                    component.plot(ax)
            else:
                self.feature_nns[0].plot()

    def plot_data(self, x, y):
        self.eval()
        with torch.no_grad():
            if len(self.feature_nns) > 1:
                fig, axes = plt.subplots(len(self.feature_nns), 1, figsize=(10, 7))
                for i, ax in enumerate(axes.flat):
                    component = self.feature_nns[i]
                    component.plot_data(ax, x[i], y)
            else:
                self.feature_nns[0].plot()


class DownstreamModel(pl.LightningModule):
    def __init__(
        self,
        trained_topic_model,
        target_column,  # Specify the name of the target column
        dataset=None,
        task="regression",
        batch_size=128,
        lr=0.0005,
    ):
        super().__init__()
        self.trained_topic_model = trained_topic_model
        self.task = task
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

        if dataset is None:
            self.structured_data = (
                self.trained_topic_model.dataset.get_structured_data()
            )
        else:
            self.structured_data = self.trained_topic_model.dataset.get_structured_data(
                data=dataset
            )
        self.target_column = target_column

        # Combine topic probabilities with structured data
        self.combined_data = self.prepare_combined_data()

        # Define the NAM architecture here based on the shape of the combined data
        self.model = self.define_nam_model()

    def prepare_combined_data(self):
        # Preprocess structured data
        preprocessed_structured_data = self.preprocess_structured_data(
            self.structured_data
        )

        # Check if soft labels are available
        if (
            hasattr(self.trained_topic_model, "soft_labels")
            and not self.trained_topic_model.soft_labels.empty
        ):
            topic_probabilities = self.trained_topic_model.soft_labels
        else:
            # Use hard labels and convert them to a one-hot encoded format
            encoder = OneHotEncoder(sparse=False)
            hard_labels = self.trained_topic_model.labels.reshape(
                -1, 1
            )  # Ensure it's a 2D array for OneHotEncoder
            topic_probabilities = pd.DataFrame(encoder.fit_transform(hard_labels))

        # Combine the preprocessed structured data with the topic probabilities
        combined_df = pd.concat(
            [preprocessed_structured_data, topic_probabilities], axis=1
        )

        # Ensure the target column is the last column in the DataFrame
        combined_df = combined_df[
            [col for col in combined_df.columns if col != self.target_column]
            + [self.target_column]
        ]

        return combined_df

    def preprocess_structured_data(self, data):
        # Identify categorical and numerical columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns

        # Define preprocessing for numerical columns (scale them)
        numerical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="mean"),
                ),  # Impute missing values with mean
                ("scaler", StandardScaler()),  # Scale numerical variables
            ]
        )

        # Define preprocessing for categorical columns (one-hot encode them)
        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),  # Impute missing values with the most frequent value
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),  # One-hot encode categorical variables
            ]
        )

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Fit and transform the data
        preprocessed_data = pd.DataFrame(preprocessor.fit_transform(data))

        # Recover the feature names generated by one-hot encoding
        feature_names = numerical_cols.tolist() + list(
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names(input_features=categorical_cols)
        )

        preprocessed_data.columns = feature_names

        return preprocessed_data

    def define_nam_model(self):
        # Define the NAM architecture
        input_size = self.combined_data.shape[1] - 1  # Exclude target column
        output_size = (
            1
            if self.task == "regression"
            else len(self.combined_data[self.target_column].unique())
        )
        model = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, output_size)
        )
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def setup(self, stage=None):
        # Split the combined data into features and target
        X = self.combined_data.iloc[:, :-1].values  # Exclude target column
        y = self.combined_data.iloc[:, -1].values

        # Convert to PyTorch tensors
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )

        # Train-validation split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
