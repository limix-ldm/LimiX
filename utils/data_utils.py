import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.inference_utils import shuffle_data_along_dim


class TabularFinetuneDataset(Dataset):
    """
        A custom PyTorch Dataset for fine-tuning, supporting data shuffling and retrieval-based selection.

        This dataset prepares training and testing splits for each item. It can either shuffle the
        training data randomly or select training examples based on pre-computed attention scores
        (retrieval). For each 'step', it provides a unique training set and a corresponding test set.
        """

    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 attention_score: np.ndarray = None,
                 retrieval_len: int = 2000,
                 use_retrieval: bool = True,
                 split_ratio: float = 0.8,
                 ):

        """
            Initializes the FinetuneDataset.
            Args:
                X_train (torch.Tensor): The full set of input training data.
                y_train (torch.Tensor): The full set of corresponding training labels.
                attention_score (np.ndarray, optional): Pre-computed attention scores for retrieval.
                                                         Shape: (num_samples_in_X_train,num_samples_in_original_X_test).
                                                         Required if use_retrieval is True.
                retrieval_len (int, optional): The number of top samples to select based on attention scores.
                                               Used only if use_retrieval is True.
                                               Note: The parameter in init_dataset is named 'train_len'.
                use_retrieval (bool, optional): Flag to determine data selection strategy.
                                                If True, uses attention scores for selection.
                                                If False, uses random shuffling.
                split_ratio (float, optional): Split ratio for selection strategy.
            """
        self.init_dataset(X_train, y_train, attention_score, retrieval_len, use_retrieval, split_ratio)

    def __len__(self):
        """
                Returns the number of steps/items in the dataset.

                Returns:
                    int: The number of steps, which corresponds to the size of the first dimension
                         of the generated X_test tensor.
                """
        return self.max_steps

    def __getitem__(self, idx: int) -> dict[str, list]:
        """
                Retrieves a single item (a training/test split configuration) by index.

                Args:
                    idx (int): The index of the item to retrieve.

                Returns:
                    dict[str, list]: A dictionary containing the tensors for the training and testing splits
                                     for this specific step/index.
                                     Keys: 'X_train', 'y_train', 'X_test', 'y_test'.
                """
        return dict(
            X_train=self.X_train[idx], # Training features for this step
            y_train=self.y_train[idx], # Training labels for this step
            X_test=self.X_test[idx], # Testing features for this step
            y_test=self.y_test[idx], # Testing labels for this step
        )

    def init_dataset(self,
                     X_train: torch.Tensor,
                     y_train: torch.Tensor,
                     attention_score: np.ndarray = None,
                     train_len: int = 2000,
                     use_retrieval: bool = False,
                     split_ratio: float = 0.8,
                     ):

        if not use_retrieval:
            X_train = shuffle_data_along_dim(X_train, 0)[:min(train_len, X_train.shape[0])]
            y_train = shuffle_data_along_dim(y_train, 0)[:min(train_len, X_train.shape[0])]
            self.X_train = torch.cat([X_train.unsqueeze(0) for _ in range(self.max_steps)], dim=0)
            self.y_train = torch.cat([y_train.unsqueeze(0) for _ in range(self.max_steps)], dim=0)
            X = self.X_train
            y = self.y_train

            # adapt train_test_split mode
            split = int(X.shape[1] * split_ratio)
            self.X_train = X[:, split:]
            self.y_train = y[:, split:]
            self.X_test = X[:, :split]
            self.y_test = y[:, :split]
            self.max_steps = self.X_test.shape[0]
        else:
            top_k_indices = np.argsort(attention_score)[:, -min(train_len, X_train.shape[0]):]
            self.X_train = torch.cat([X_train[x_iter].unsqueeze(0) for x_iter in top_k_indices], dim=0)
            self.y_train = torch.cat([y_train[x_iter].unsqueeze(0) for x_iter in top_k_indices], dim=0)
            X = shuffle_data_along_dim(self.X_train, 1)
            y = shuffle_data_along_dim(self.y_train, 1)

            # adapt train_test_split mode
            split = int(X.shape[1] * split_ratio)
            self.X_train = X[:, split:]
            self.y_train = y[:, split:]
            self.X_test = X[:, :split]
            self.y_test = y[:, :split]
            self.max_steps = self.X_train.shape[0]


class TabularInferenceDataset(Dataset):
    """
        A PyTorch Dataset for tabular data inference scenarios.

        This dataset is designed to provide data for inference tasks where
        you might have a fixed training set and varying test samples, optionally
        selecting the training set based on relevance (retrieval) for each test sample.
        When retrieval is used, each test sample (or step) is paired with a specific,
        potentially unique, subset of the training data. When retrieval is not used,
        it's assumed a single, fixed training set is used for all test samples.
        """

    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 X_test: torch.Tensor,
                 attention_score: np.ndarray|torch.Tensor = None,
                 retrieval_len: int = 2000,
                 use_retrieval: bool = True,
                 ):
        """
                Initializes the TabularInferenceDataset.

                Args:
                    X_train (torch.Tensor): The full set of input training features.
                                            Shape: (num_train_samples, ...).
                    y_train (torch.Tensor): The full set of corresponding training labels.
                                            Shape: (num_train_samples, ...).
                    X_test (torch.Tensor): The set of input features for inference/test samples.
                                           Shape: (num_test_samples, ...).
                    attention_score (np.ndarray, optional): Pre-computed attention scores
                                                            for retrieval logic. Shape depends
                                                            on implementation, e.g., Shape: (num_samples_in_X_train,num_samples_in_X_test).
                                                            Required if use_retrieval is True.
                    retrieval_len (int, optional): The number of top training samples to select
                                                   based on attention scores for each test sample.
                                                   Used only if use_retrieval is True.
                    use_retrieval (bool, optional): Flag to determine data preparation strategy.
                                                    If True, uses attention scores to select relevant training data
                                                    for each test sample.
                                                    If False, assumes a fixed training set is used for all.
                """
        self.init_dataset(X_train, y_train, X_test, attention_score, retrieval_len, use_retrieval)
        # The number of inference steps equals the number of test samples
        self.max_steps = self.X_test.shape[0]
        self.use_retrieval = use_retrieval

    def __len__(self):
        """
                Returns the number of steps/items in the dataset.
                Returns:
                    int: The number of steps, which corresponds to the size of the first dimension
                         of the generated X_test tensor.
                """
        return self.max_steps

    def __getitem__(self, idx: int) -> dict[str, list]:
        """
                Retrieves a single item (data for one inference step) by index.

                Args:
                    idx (int): The index of the test sample/step to retrieve.

                Returns:
                    dict[str, torch.Tensor]: A dictionary containing the data needed for this inference step.
                                             If `use_retrieval` is True, it includes the specific
                                             `X_train`, `y_train`, and `X_test` for this step.
                                             If `use_retrieval` is False, it only includes `X_test`,
                                             as a fixed training set is assumed.
                """
        if self.use_retrieval:
            # Return the specific training data selected for this test sample
            return dict(
                idx=int(idx),
                X_train=self.X_train[idx], # Training features for this step (retrieved)
                X_test=self.X_test[idx], # Training labels for this step (retrieved)
                y_train=self.y_train[idx], # The test sample features
            )
        else:
            # Return only the test data; training data is assumed to be fixed and
            # provided.
            return dict(
                idx=int(idx),
                X_test=self.X_test[idx],
            )

    def init_dataset(self,
                     X_train: torch.Tensor,
                     y_train: torch.Tensor,
                     X_test: torch.Tensor,
                     attention_score: np.ndarray = None,
                     train_len: int = 2000,
                     use_retrieval: bool = False,
                     ):
        if use_retrieval:
            top_k_indices = np.argsort(attention_score)[:, -min(train_len, X_train.shape[0]):]
            self.X_train = torch.cat([X_train[x_iter].unsqueeze(0) for x_iter in top_k_indices], dim=0)
            self.y_train = torch.cat([y_train[y_iter].unsqueeze(0) for y_iter in top_k_indices], dim=0).unsqueeze(-1)
            self.X_test = X_test
        else:
            self.X_test = X_test




def load_data(data_root,folder):
    le = LabelEncoder()
    train_path = os.path.join(data_root,folder, folder + '_train.csv')
    test_path = os.path.join(data_root,folder, folder + '_test.csv')
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
        else:
            train_df, test_df = train_test_split(train_df, test_size=0.5, random_state=42)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            try:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
            except Exception as e:
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    trainX, trainy = X_train, y_train
    trainX = np.asarray(trainX, dtype=np.float32)
    trainy = np.asarray(trainy, dtype=np.int64)


    testX, testy = X_test, y_test
    testX = np.asarray(testX, dtype=np.float32)
    testy = np.asarray(testy, dtype=np.int64)
    return trainX, trainy, testX, testy
if __name__ == '__main__':
    pass

