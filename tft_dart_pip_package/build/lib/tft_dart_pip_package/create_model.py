import warnings

warnings.filterwarnings("ignore")

import os
import time
import random
import pandas as pd
import pickle
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from itertools import product
import torch
from torch import nn
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.utils.losses import SmapeLoss
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape
from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
from darts.models import *
from pandas.tseries.frequencies import to_offset
from darts.utils.likelihood_models import QuantileRegression
import torch
import json

from preprocess import getTimeSeries
from darts.metrics import mape


class load_and_train_model:
    def __init__(self):
        self.config_path = 'model_config.json'
        self.config = self.load_config(self.config_path)

    def load_config(self,config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config

    def parallelize_model(self,model):
        if torch.cuda.device_count() > 1:
            print("using multiple", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f'found {device}')
            model.to(device)
        return model
    
    def load_model(self):
        # default quantiles for QuantileRegression
        quantiles = [
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ]

        my_model = TFTModel(
            input_chunk_length=self.config['input_len'],
            output_chunk_length=self.config['horizon_len'],
            hidden_size=self.config['hidden_size'],
            lstm_layers=self.config['lstm_layers'],
            num_attention_heads=self.config['num_attention_heads'],
            dropout=self.config['dropout'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            add_relative_index=False,
            add_encoders=None,
            likelihood=QuantileRegression(
                quantiles=quantiles
            ),  # QuantileRegression is set per default
            # loss_fn=MSELoss(),
            random_state=42,
        )
        # my_model = self.parallelize_model(my_model)
        return my_model
    
    def train_model(self):
        cls = getTimeSeries(split_point = 0.7)
        train,eval,cov_transformed = cls.create()

        my_model = self.load_model()
        my_model.fit(train, future_covariates = cov_transformed, verbose=True)

        whole_series = cls.converted_series
        self.eval_model(model = my_model,n = 10, actual_series = whole_series,
                        val_series = eval)

    def eval_model(self, model, n, actual_series, val_series):
        num_samples = 200
        figsize = (9, 6)
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
        label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
        label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

        pred_series = model.predict(n=n, num_samples = 10)

        # plot actual series
        plt.figure()
        actual_series[: pred_series.end_time()].plot(label="actual")

        # plot prediction with quantile ranges
        pred_series.plot(
            low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
        )
        pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
        MAPE_score = mape(val_series, pred_series)
        print("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
        plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
        plt.legend()
