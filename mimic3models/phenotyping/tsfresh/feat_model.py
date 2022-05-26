import numpy as np                                                                   
from feat import FeatClassifier

est = FeatClassifier(
           # classification=True,
           max_depth=6,                                                              
           max_dim = 60,
           obj='fitness,complexity',
           sel='lexicase',
           surv='nsga2',
           gens = 10,
           # pop_size = 2000,
           pop_size = 500,
           max_stall = 100,
           # stagewise_xo = True,
           scorer='log',
           verbosity=2,
           ml='L2_LR',
           fb=0.1,
           # n_threads=4,
           # n_threads=20,
           n_jobs=20,
           # functions= "split,and,or,not,b2f",
           functions= ("+,-,*,/,^2,sqrt,exp,log,"
                  "relu,"
                  "split,"
                  # "split,split_c,fuzzy_split,fuzzy_split_c,"
                  # "fuzzy_fixed_split,fuzzy_fixed_split_c,"
                  "b2f,and,or,not,=,<,<=,>,>=" #,if,ite"
                  ),
           split=0.8,
           shuffle=False, #this preserves the train/val sets from yerevan
           normalize=True,
           corr_delete_mutate=True, 
           # simplify=0.005,
           batch_size=512,
           backprop=True,
           iters=1,
           tune_initial=False,
           tune_final=True
           ) 

hyper_params = {}
