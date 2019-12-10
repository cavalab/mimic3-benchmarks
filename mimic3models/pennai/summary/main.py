from __future__ import absolute_import
from __future__ import print_function

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from mimic3benchmark.readers import PhenotypingReader
from mimic3models import common_utils
from mimic3models import metrics
from mimic3models.phenotyping.utils import save_results

import numpy as np
import pandas as pd
import argparse
import os
import json


def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    #ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_pennai_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'], ret['t'], ret['header'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--grid-search', dest='grid_search', action='store_true')
    parser.add_argument('--no-grid-search', dest='grid_search', action='store_false')
    parser.set_defaults(grid_search=False)
    parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/phenotyping/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    if args.grid_search:
        penalties = ['l2', 'l2', 'l2', 'l2', 'l2', 'l2', 'l1', 'l1', 'l1', 'l1', 'l1']
        coefs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001, 0.0001]
    else:
        penalties = ['l1']
        coefs = [0.1]

    train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                     listfile=os.path.join(args.data, 'train_listfile.csv'))

    val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                   listfile=os.path.join(args.data, 'val_listfile.csv'))

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

    print('Reading data and extracting features ...')

    (train_X, train_y, train_names, train_ts, train_header) = read_and_extract_features(train_reader, args.period, args.features)
    train_y = np.array(train_y)

    (val_X, val_y, val_names, val_ts, val_header) = read_and_extract_features(val_reader, args.period, args.features)
    val_y = np.array(val_y)

    (test_X, test_y, test_names, test_ts, test_header) = read_and_extract_features(test_reader, args.period, args.features)
    test_y = np.array(test_y)

    header = []
    if(train_header != test_header):
        print("something went wrong.  training and test headers do not match")
        exit()
    for i in train_header: 
        if(i != 'Hours'):
            for stat in ['min','max', 'mean', 'median', 'std', 'autocorr1', 'autocorr2', 'autocorr3', 'autocorr4', 'autocorr5' ]:
                if((stat == 'mean' and i not in ['Capillary refill rate','Fraction inspired oxygen','Height']) or i in ['Diastolic blood pressure','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure']):
                    header.append(i + '_' + stat) 
                else:
                    header.append('deleteme_' + i + '_' + stat) 
    
    header.append('class') 
    np.save('/tmp/preimputed_train_X',train_X)

    print("train set shape:  {}".format(train_X.shape))
    print("validation set shape: {}".format(val_X.shape))
    print("test set shape: {}".format(test_X.shape))

    print('Imputing missing values ...')
    imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)
    train=np.append(train_X,train_y[:,23][:, None],axis=1)




    summary=pd.DataFrame(data=train,columns=header)
    summary.to_pickle(args.output_dir + "./summary.pkl")
    class_count=min(summary.groupby('class').size())
    summary=summary.assign(class_count=0)
    count_true = 0
    count_false = 0
    for i, row in summary.iterrows():
     if (row['class'] == 1):
       summary.set_value(i,'class_count',count_true)
       count_true+=1
     if (row['class'] == 0):
       summary.set_value(i,'class_count',count_false)
       count_false+=1

    summary = summary[summary.class_count < class_count]
    summary = summary[summary.columns.drop(list(summary.filter(regex='deleteme')))]
    summary = summary.drop(columns=['class_count'])

    summary = summary.round(decimals=3)
    summary.to_csv(args.output_dir + './summary.csv',index=False)
    
    np.savetxt(args.output_dir + '/pennai.csv',train, delimiter=',', fmt='%1.2f')

if __name__ == '__main__':
    main()
