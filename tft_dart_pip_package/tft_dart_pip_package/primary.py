from create_model import *
from preprocess import *
import torch
from darts.metrics import mape

cls = load_and_train_model() 
cls.train_model()