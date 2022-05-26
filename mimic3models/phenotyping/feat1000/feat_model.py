import numpy as np                                                                   
from feat import FeatClassifier


est = FeatClassifier(
        max_depth=3,
        max_dim = 1000,
        obj='fitness,size',
        sel='lexicase',
        # gens = 0,
        # pop_size = 1,
        # tune_final=False,
        gens = 100,
        pop_size = 100,
        max_stall = 20,
        stagewise_xo = True,
        scorer='log',
        verbosity=1,
        shuffle=False,
        ml='LR',
        fb=0.5,
        n_jobs=1,
        functions="split,and,or,not,b2f",
        split=0.8,
        normalize=False,
        corr_delete_mutate=True,
        simplify=0.005
) 

hyper_params = [
    dict(
         functions="split,and,or,not,b2f"
        ),
    dict(
         functions="+,-,*,relu,split",
         batch_size=200,
         backprop=True,
         iters=1,
        )
]
