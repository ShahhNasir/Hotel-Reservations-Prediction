from scipy.stats import randint,uniform

LIGHTGM_PARAMS = {
    'n_estimators':randint(100,500),
    'max_depth': randint(5,50),
    'learning_rate': uniform(0.01,0.02),
    'num_leaves':[15,31,63],
    'boosting_type': ['goss']

}


RANDOM_SEARCH_PARAMS = {
    'n_iter' : 4,
    'cv' : 2,
    'n_jobs' : -1,
    'verbose' : 2,
    'random_state' : 33,
    'scoring' : 'accuracy'
}