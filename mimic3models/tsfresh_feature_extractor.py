from __future__ import absolute_import
from __future__ import print_function
import pdb
# import ipdb
import numpy as np
from scipy.stats import skew
import pandas as pd
import math
from pqdm.processes import pqdm
from tqdm import tqdm
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
 

n_functions = len(ComprehensiveFCParameters()) 

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4 * 24),
    "first8days": (0, 0, 0, 8 * 24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50)
}
def make_windows(data):
    """make windows for extracting time series"""
    first4 = data.loc[data.Hours <= 4*24,:]
    first8 = data.loc[data.Hours <= 8*24,:]

    print('last 12..')
    hrs_max = data.groupby('PID', as_index=False)['Hours'].max()
    tmp = data.merge(hrs_max, on='PID', suffixes=('','_max'))
    last12 = (tmp.loc[tmp['Hours_max'] - tmp['Hours'] <= 12,:]
              .drop('Hours_max', axis=1)
             )
    return [(data,'all'), 
            (first4,'first4days'), 
            (first8,'first8days'), 
            (last12,'last12hours')]
               

sub_periods = [(2, 100)]



def get_range(begin, end, period):
    # first p %
    if period[0] == 2:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p %
    if period[0] == 3:
        return (end - (end - begin) * period[1] / 100.0, end)

    if period[0] == 0:
        L = begin + period[1]
    else:
        L = end + period[1]

    if period[2] == 0:
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period):
    if len(channel_data) == 0:
        return np.full((n_functions), np.nan)

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data
            if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((n_functions), np.nan)
        # return np.full((len(functions, )), np.nan)
    # return np.array([fn(data) for fn in functions], dtype=np.float32)
    return extract_features(data)


def extract_features_single_episode(data_raw, period): #, functions):
    global sub_periods
    pdb.set_trace()
    extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)

from tqdm import tqdm
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

import errno
def extract(timeseries_container=None, column_id='PID', column_sort='Hours',
            column_kind='variable', column_value='value', n_jobs=0):
    try:
        return extract_features(
                         timeseries_container=timeseries_container, 
                         column_id=column_id,
                         column_sort=column_sort,
                         column_kind=column_kind,
                         column_value=column_value,
                         n_jobs=n_jobs
                        )
    # except IOError as e:
    except Exception as e:
        print(e)
        # if e.errno == errno.EPIPE:
        return None


def extract_tsfresh_features(data_raw, period, fold, features):
    chunksize = 50
    data = pqdm([dict(timeseries_container=pd.concat(X), n_jobs=0)
                 for X in chunker(data_raw, chunksize)],
                extract,
                n_jobs=10,
                argument_type='kwargs'
                )
    # ipdb.set_trace()
    dataset= pd.concat([d for d in data if isinstance(d, pd.DataFrame)])
    dataset.to_csv(f'tsfresh_extracted_features_{fold}.csv',index=False)
    return dataset
