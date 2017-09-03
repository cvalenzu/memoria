import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import datetime
from multiprocessing import Pool
import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def find_missing(serie):
    dates = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq="H")
    missing_dates = []

    continous_part = []

    start = serie.index.min()
    end = serie.index.min()
    for date in dates:
        if not date in serie.index:
            missing_dates.append(date)
            continous_part.append((start,end))
            start = date
            end = date
        else:
            end = date

    if len(missing_dates) == 0:
        print("No missing values")
        return serie.index.min(), serie.index.max()


    missing = pd.DataFrame(missing_dates)
    missing.index = missing[0]

    length = []
    for start,end in continous_part:
        length.append((end - start).days)
    best_start,best_end = continous_part[np.argmax(length)]
    print(best_start,best_end)

    return best_start,best_end

def train_arima(args):
    data,test,params,i = args
    model = sm.tsa.ARIMA(data,params)
    result = model.fit()
    pred = result.forecast(steps=12)
    return pred[0]

parser = argparse.ArgumentParser()
parser.add_argument("--nproc","-n",help="Number of processors",type=int,default=1)
args = parser.parse_args()

file = "lota_r.csv"
print("Loading Data [",file,"]")
data = pd.read_csv('../../../data/'+file, names=["t","ws"],index_col="t")
data.index=pd.to_datetime(data.index)
print("Filtering Data")
start,end = find_missing(data)
data_filtered = data[start:end]

print("Dividing data")
data_len = len(data_filtered)
train_perc = 0.9
train_data = int(0.9*data_len)
train = data_filtered[:train_data]
test = data_filtered[train_data:]

print("Creating workers and training")
pool = Pool(2)

parameters = (3,1,3)
data = train
wind_speed = []

args = []
for i,value in enumerate(test.iterrows()):
    if i == len(test)- 12:
        break
    args.append((data,test,parameters,i))
    wind_speed.append(test[i:i+12].values.reshape((-1,)))

    index, val = value
    data = data.append(val)

forecast = np.array(pool.map(train_arima,args))
wind_speed = np.array(wind_speed)
print("Workers done, saving results to files")

np.savetxt("y_pred.txt",forecast)
np.savetxt("y_test.txt",wind_speed)
