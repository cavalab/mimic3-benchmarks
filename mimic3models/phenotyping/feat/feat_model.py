import numpy as np                                                                   
from feat import FeatClassifier


est = FeatClassifier(
        max_depth=6,
        max_dim = 20,
        obj='fitness,size',
        sel='lexicase',
        gens = 1,
        pop_size = 1,
        max_stall = 20,
        stagewise_xo = True,
        scorer='log',
        verbosity=2,
        shuffle=False,
        ml='LR',
        fb=0.5,
        n_jobs=20,
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
         iters=1
        )
]
