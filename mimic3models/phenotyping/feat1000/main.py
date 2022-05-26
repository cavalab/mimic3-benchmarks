from __future__ import absolute_import
from __future__ import print_function

from sklearn.model_selection import ParameterGrid
from mimic3benchmark.readers import PhenotypingReader
from mimic3models import common_utils
from mimic3models import metrics
# from mimic3models.phenotyping.utils import save_results
from .feat_model import est, hyper_params
# from mimic3models.pennai_feature_extractor 

import pandas as pd
import numpy as np
import argparse
import os
import json
import ipdb
import uuid
from .jsonify import jsonify

from mimic3models.common_utils import phenotype_names as phenotype_names
phenotype_names = [p.replace(' ','-').replace(';','') for p in phenotype_names]

def read_data(datapath, features, phenotype, fold):
    if features == 'extract':
        dropcols = phenotype_names
        filepath = f'{datapath}/{features}/{fold}.csv'
        if phenotype == 'all':
            ynames = phenotype_names
        else:
            ynames = phenotype
            assert phenotype in phenotype_names
    elif features == 'tsfresh':
        filepath = f'{datapath}/{features}/{phenotype}_{fold}.csv'
        dropcols = 'class'
        ynames = 'class'
    print('reading',filepath)
    df = pd.read_csv(filepath).drop("Unnamed: 0", axis=1)
    df = df.rename(columns={k:k.replace('"','').strip() for k in df.columns})
    X = df.drop(dropcols, axis=1)
    renames = {k:k.replace(',','_').replace(' ','_') for k in X.columns}
    X = X.rename(columns=renames)
    print('Features:',X.columns)

    y = pd.DataFrame(df[ynames])
    if features == 'tsfresh':
        y = y.rename(columns={'class':phenotype})

    return (X, y)

def save_results(task, predictions, labels, path):

    common_utils.create_directory(os.path.dirname(path))
    
    results = dict(
                task=task,
                pred=predictions.tolist(),
                label=labels.values.tolist()
              )
    with open(path, 'w') as f:
        json.dump(results, f)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--period', type=str, default='all', 
    #         help='specifies which period extract features from',
    #         choices=['first4days', 'first8days', 'last12hours', 
    parser.add_argument('--seed', type=int, default=42,
                        help='random state')
    parser.add_argument('--features', type=str, default='extract',
                        help='specifies which feature set to use',
                        choices=['extract','tsfresh'])
    parser.add_argument('--phenotype', type=str, help='which phenotype to model', 
                        default='all', choices=['all']+phenotype_names)
    parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/phenotyping/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    args.phenotype=args.phenotype.replace('"','').strip()
    print('Reading data ...')

    (train_X, train_y) = read_data(args.data,
                                   args.features,
                                   args.phenotype,
                                   'train'
                                  )
    (val_X, val_y) = read_data(args.data, 
                               args.features,
                               args.phenotype, 
                               'val'
                              )
    trainval_X = pd.concat((train_X, val_X))
    trainval_y = pd.concat((train_y, val_y))
    assert trainval_X.shape[0] == len(train_X) + len(val_X)
    assert trainval_X.shape[1] == train_X.shape[1] 
    # set FEAT split to use training/validation appropriately
    est.split = float(len(train_X)/len(trainval_X))
    assert est.shuffle == False

    (test_X, test_y) = read_data(args.data, 
                                 args.features,
                                 args.phenotype, 
                                 'test'
                                )

    print("train set shape: {}".format(train_X.shape))
    print("validation set shape: {}".format(val_X.shape))
    print("test set shape: {}".format(test_X.shape))

    if args.phenotype == 'all':
        tasks = phenotype_names
    else:
        assert args.phenotype in phenotype_names
        tasks = [args.phenotype]

    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    logp = ['pop_size','gens','ml','backprop'] 
    # for param_id, params in enumerate(ParameterGrid(hyper_params)):
    for param_id, params in enumerate(hyper_params):
        est.set_params(**params)
        est.random_state = args.seed
        run_id = uuid.uuid1().hex
        model_name = f'feat.dim1000.run_{run_id}.param_{param_id}'

        train_activations = np.zeros(shape=train_y.shape, dtype=float)
        val_activations = np.zeros(shape=val_y.shape, dtype=float)
        test_activations = np.zeros(shape=test_y.shape, dtype=float)

        for task_id, task in enumerate(list(tasks)):
            print('Starting task {}'.format(task))

            # logreg = LogisticRegression(penalty=penalty, C=C, random_state=42)
            est.fit(trainval_X, trainval_y[task])
            train_preds = est.predict_proba(train_X)
            train_activations[:, task_id] = train_preds[:, 1]

            val_preds = est.predict_proba(val_X)
            val_activations[:, task_id] = val_preds[:, 1]

            test_preds = est.predict_proba(test_X)
            test_activations[:, task_id] = test_preds[:, 1]
        
            save_results(task, 
                         test_activations[:, task_id],
                         test_y[task],
                         os.path.join(args.output_dir, 
                                      'predictions', 
                                      model_name + '.json'
                                     )
                         )
            ret={}
            ret['data'] = args.features
            ret['seed'] = args.seed
            ret['task'] = task
            ret['run_id'] = run_id
            ret['param_id'] = param_id
            ret['fold'] = 'train'
            ret['method'] = 'FEAT-1000'
            ret['model'] = est.get_eqn()
            ret['n_nodes'] = est.get_n_nodes()
            ret['metrics'] = {'train':{},'val':{},'test':{}}
            frames = []
            for y_true, y_pred, fold in [
                                         (train_y[task], train_preds, 'train'),
                                         (val_y[task], val_preds, 'val'),
                                         (test_y[task], test_preds, 'test')
                                        ]:
                ret['fold'] = fold
                emtrix = metrics.print_metrics_binary(y_true, y_pred)
                for k,v in emtrix.items():
                    result = ret.copy()
                    result['metric'] = k
                    result['value'] = v
                    frames.append(result)

            results = pd.DataFrame.from_records(frames)
            results.to_csv(os.path.join(result_dir,f'{task}.{model_name}.csv'),
                           index=False)

            with open(os.path.join(result_dir, 
                      f'{task}.{model_name}.params'), 'w') as f:
                jsonparams = jsonify(est.get_params())
                json.dump(jsonparams,f)

if __name__ == '__main__':
    main()
