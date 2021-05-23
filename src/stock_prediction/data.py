import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

import torch
import pytorch_lightning as pl

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader



class TimeSeriesDataset(Dataset):
    def __init__(self, datasets):
        super(TimeSeriesDataset, self).__init__()
        self._length = len(datasets[0])
        for i, data in enumerate(datasets):
            assert len(data) == self._length, \
                "All arrays must have the same length; \
                array[0] has length %d while array[%d] has length %d." \
                % (self._length, i+1, len(data))
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return (torch.tensor(self.datasets[0][idx], dtype=torch.float32),
                torch.tensor(self.datasets[1][idx], dtype=torch.float32))


class StocksDataModule(pl.LightningDataModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str,
                            help="Path to data.")
        parser.add_argument("--batch_size", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--num_workers", type=int, default=8,
                            help="Number of workers for data loading.")
        parser.add_argument("--special_tokens", nargs="*", default=["[CON]", "[QUE]", "[ANS]", "[DIS]"],
                            help="Additional special tokens.")
        parser.add_argument("--pretrained_model", type=str, default="prajjwal1/bert-tiny",
                            help="Pretrained model.")
        return parser
    
    @staticmethod
    def split_sequence(source, target, fea_width, label_width,
                        step, target_start_next):
        """ Split sequence with sliding window into
            sequences of context features and target.
            Args:
                source (np.array): Source sequence
                target (np.array): Target sequence
                fea_width (int): Length of input sequence.
                label_width (int): Length of target sequence.
                target_start_next (bool): If True, target sequence
                        starts on the next time step of last step of source
                        sequence. If False, target sequence starts at the
                        same time step of source sequence.
            Return:
                X (np.array): sequence of features
                y (np.array): sequence of targets
        """

        X, y = list(), list()

        if not target_start_next:
            target = np.vstack((np.zeros(target.shape[1],
                                            dtype=target.dtype), target))
        for i in range(0, len(source), step):
            # Find the end of this pattern:
            src_end = i + fea_width
            tgt_end = src_end + label_width
            # Check if beyond the length of sequence:
            if tgt_end > len(target):
                break
            # Split sequences:
            X.append(source[i:src_end, :])
            y.append(target[src_end:tgt_end, :])
        return np.array(X), np.array(y)
    
    def __init__(
        self, scaler=StandardScaler(), src_cols=["Sentiment_Change","Price_Change"],
        tgt_cols=["Price_Change"], fea_width = 4, label_width = 1, 
        step=1, target_start_next=True, classification = False, num_workers=2
    ):
        
        super().__init__()
        self.X_scaler, self.y_scaler = scaler,scaler
        
        self.src_cols = src_cols
        self.tgt_cols = tgt_cols
        self.fea_width = fea_width
        self.label_width = label_width
        self.step = step
        self.target_start_next = target_start_next
        self.num_workers = num_workers
        self.classification = classification
        

    def prepare_data(self):
        self.df = pd.read_csv("project/data/tagged.csv", parse_dates=[0])
        self.src, self.tgt = self.df[self.src_cols].values, \
                    self.df[self.tgt_cols].values

    def setup(self):
        
        # Split data into training set and val_test set :
        X_train, X_val_test, y_train, y_val_test \
            = train_test_split(self.src, self.tgt,
                               test_size=0.3, shuffle=False)
        
        if not self.classification:
            y_train = self.y_scaler.fit_transform(y_train)
            y_val_test = self.y_scaler.fit_transform(y_val_test)

        X_train = self.X_scaler.fit_transform(X_train)
        X_val_test = self.X_scaler.transform(X_val_test)
        
        X_train, y_train = self.split_sequence(
                X_train, y_train, self.fea_width,
                self.label_width,self.step, self.target_start_next
        )

        X_val_test, y_val_test = self.split_sequence(
                X_val_test, y_val_test, self.fea_width,
                self.label_width,self.step, self.target_start_next
        )

        # Split training data into validation set and test set:
        X_val, X_test, y_val, y_test \
            = train_test_split(X_val_test, y_val_test,
                               test_size=0.5, shuffle=False)


        # Prepare datasets
        self.trainset = TimeSeriesDataset([X_train, y_train])
        self.valset = TimeSeriesDataset([X_val, y_val])
        self.testset = TimeSeriesDataset([X_test, y_test])

    def train_dataloader(self):
        self.train_loader = DataLoader(
                self.trainset,
                batch_size=len(self.trainset),
                shuffle=False,
                num_workers=self.num_workers
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(
                self.valset,
                batch_size=len(self.valset),
                shuffle=False,
                num_workers=self.num_workers
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
                self.testset,
                batch_size=len(self.testset),
                shuffle=False,
                num_workers=self.num_workers
        )
        return self.test_loader