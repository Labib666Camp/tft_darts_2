import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

global global_path
global_path = 'LD2011_2014.txt'

class getTimeSeries:
    def __init__(self,split_point):
        self.path = global_path
        self.df = None
        self.split_point = split_point
        self.converted_series = None
        self.load()

    def load(self):
        print(f'-----loading data-------')
        self.df = pd.read_csv(
            "LD2011_2014.txt",
            sep=";",
            index_col=0,
            parse_dates=True,
            decimal=",",
        )
        self.df.index = pd.to_datetime(self.df.index)
        print(f'------data loaded---------')

    def create(self):
        cols = self.df.columns
        converted_series = []
        covariaate_series = []
        train_series = []
        val_series = []
        time_ids = self.df.index

        i = 0
        print(f'creating TimeSeries objects ------')
        for col in cols:
            # print(f'{col} starting -----')
            series = TimeSeries.from_series(self.df[col])
            transformer = Scaler()
            series_transformed = transformer.fit_transform(series)
            converted_series.append(series_transformed)

            # training_cutoff = pd.Timestamp("20130101")
            train,val = series_transformed.split_after(self.split_point)
            train_series.append(train)
            val_series.append(val)

            cov = self.covariates_single_series(series)
            covariaate_series.append(cov)

            i = i + 1

            if (i == 10):
                break
        print(f'TimeSeries Creation DONE ------')
        self.converted_series = concatenate(converted_series, axis=2)
        train_series = concatenate(train_series, axis=2)
        val_series = concatenate(val_series, axis=2)
        
        cov_transformed = concatenate(covariaate_series,axis = 2)

        print(f'all {self.converted_series.data_array().shape}')
        print(f'train {train_series.data_array().shape}')
        print(f'val {train_series.data_array().shape}')
        print(f'cov {cov_transformed.data_array().shape}')

        return train_series,val_series,cov_transformed

    def covariates_single_series(self,series):
        covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
        covariates = covariates.stack(
            datetime_attribute_timeseries(series, attribute="month", one_hot=False)
        )
        covariates = covariates.stack(
            TimeSeries.from_times_and_values(
                times=series.time_index,
                values=np.arange(len(series)),
                columns=["linear_increase"],
            )
        )
        covariates = covariates.astype(np.float32)
        # forecast_horizon_ice = 12
        # training_cutoff = series.time_index[-(2 * forecast_horizon_ice)]
        
        
        # transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
        # training_cutoff = pd.Timestamp('20130101')
        scaler_covs = Scaler()
        cov_train, cov_val = covariates.split_after(self.split_point)
        scaler_covs.fit(cov_train)
        covariates_transformed = scaler_covs.transform(covariates) 
        return covariates_transformed