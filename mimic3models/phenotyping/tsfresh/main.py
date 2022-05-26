from __future__ import absolute_import

from __future__ import print_function

import ipdb
import pandas as pd
# try:
#     from sklearn.preprocessing import Imputer, StandardScaler
# except Exception as e:
#     print(e)
#     from sklearn.impute import SimpleImputer as Imputer 
# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import ParameterGrid
# from sklearn.linear_model import LogisticRegression
from mimic3benchmark.readers import PhenotypingReader
from mimic3models import common_utils
from mimic3models import metrics
from mimic3models.phenotyping.utils import save_results
# from .feat_model import est, hyper_params
# from mimic3models.pennai_feature_extractor 
# from mimic3models.tsfresh_feature_extractor 
from tsfresh import extract_relevant_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute


import numpy as np
import argparse
import os
import json


def read_and_extract_features(reader, period, fold, features=None):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    # if os.path.exists(f'sm_tsfresh_extracted_features_{fold}.csv'):
    #     print('reading csv...')
    #     X = pd.read_csv(f'sm_tsfresh_extracted_features_{fold}.csv',index_col=False)
    if os.path.exists(f'tsfresh_extracted_features_{fold}.csv'):
        print('reading csv...')
        X = pd.read_csv(f'tsfresh_extracted_features_{fold}.csv',index_col=False)
        # ipdb.set_trace()
    else:
        X = common_utils.extract_tsfresh_from_rawdata(ret['X'], ret['header'],
                period, features)

    idx = X['Unnamed: 0'].values
    y = np.array(ret['y'])[idx,:]

    return (X, y, ret['name'], ret['t'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.set_defaults(grid_search=False)
    parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/phenotyping/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)



    train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                     listfile=os.path.join(args.data, 'train_listfile.csv'))
                                     # listfile=os.path.join(args.data, 'sm_train_listfile.csv'))


    val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                   listfile=os.path.join(args.data, 'val_listfile.csv'))
                                   # listfile=os.path.join(args.data, 'sm_val_listfile.csv'))

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))
                                    # listfile=os.path.join(args.data, 'sm_test_listfile.csv'))


    print('Reading data and extracting features ...')

    (train_X, train_y, train_names, train_ts) = \
            read_and_extract_features(train_reader, args.period, 'train' )
    train_y = np.array(train_y)

    (val_X, val_y, val_names, val_ts) = read_and_extract_features(val_reader,
            args.period, 'val', features='all')
    val_y = np.array(val_y)

    (test_X, test_y, test_names, test_ts) = \
        read_and_extract_features(test_reader, args.period, 'test', features='all')
    test_y = np.array(test_y)



    print("train set shape:  {}".format(train_X.shape))
    print("validation set shape: {}".format(val_X.shape))
    print("test set shape: {}".format(test_X.shape))

    # print('Imputing missing values ...')
    # imputer = Imputer(missing_values=np.nan, strategy='median', axis=0, verbose=0, copy=True)
    # imputer.fit(train_X)
    # train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    # val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    # test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    # print('Normalizing the data to have zero mean and unit variance ...')
    # scaler = StandardScaler()
    # scaler.fit(train_X)
    # train_X = scaler.transform(train_X)
    # val_X = scaler.transform(val_X)
    # test_X = scaler.transform(test_X)

    n_tasks = 25
    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)
    
    outcomes = train_reader._listfile_header.split(',')[2:]
    outcomes = [o.replace('"','').strip() for o in outcomes]

    impute(train_X)
    impute(val_X)
    impute(test_X)

    # determine relevant features for each target y
    for (i,outcome) in enumerate(outcomes):
        trainy_X = select_features(train_X, train_y[:,i])
        selected_features = trainy_X.columns
        print(len(selected_features),'features selected for',outcome)

        valy_X = val_X[selected_features]
        testy_X = test_X[selected_features]

        for t,x,y in [('train',trainy_X, train_y[:,i]), 
                      ('val', valy_X, val_y[:,i]), 
                      ('test', testy_X, test_y[:,i])
                     ]:
        # columns = [f'x{i}' for i in range(x.shape[1])] + outcomes
            # data = np.hstack((x,y.reshape(-1,1)))
            # assert data.shape[0] == len(x) 
            # assert data.shape[1] == x.shape[1] + 1

            data = x
            data.loc[:,'class'] = y
            # df = pd.DataFrame(data, 
            #         columns=columns)
            # df = pd.DataFrame(data, 
            #         columns=train)
            # ipdb.set_trace()
            out = outcome.replace(' ','-')
            savefile = os.path.join(args.output_dir,
                                      'tsfresh',
                                      f'{out}_{t}.csv')

            data.to_csv(savefile)
            print(savefile,'written')
        
         
    # for t, x, y, reader in [('train', train_X, train_y, train_reader),
    #                         ('val', val_X, val_y, val_reader),
    #                         ('test', test_X, test_y, test_reader)
    #                         ]:
    #     assert train_y.shape[1] == len(outcomes)
    #     columns = [f'x{i}' for i in range(x.shape[1])] + outcomes
    #     data = np.hstack((x,y))
    #     assert data.shape[0] == len(x) 
    #     assert data.shape[1] == x.shape[1] + y.shape[1]

    #     df = pd.DataFrame(data, 
    #             columns=columns)
    #     df.to_csv( os.path.join(args.output_dir,'tsfresh',f'{t}.csv'))
    

if __name__ == '__main__':
    main()
