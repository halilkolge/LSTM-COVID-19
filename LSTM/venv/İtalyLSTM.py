import itertools
from math import sqrt
from datetime import datetime
from numpy import concatenate
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.recurrent import LSTM
from sklearn.utils import shuffle
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv('italy_data.csv')
df['TotalPositiveCases']= df['TotalPositiveCases'] -( df['Recovered']+ df['Deaths'] )
df=df.groupby(['Date']).sum()
FilteredDf=df.filter(items=['TotalPositiveCases','Date'])
data= FilteredDf.reset_index()['TotalPositiveCases']


print(data)