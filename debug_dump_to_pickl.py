from psycopt2d.utils import dump_to_pickle

if __name__ == "__main__":
    obj = 2

    PATH_TO_DUMP = "E:\\shared_resources\\model_predictions\\psycop-t2d-testing\\eval_model_name-xgboost_require_imputation-True_args-n_estimators-800_tree_method-auto_lambda-0.7016387837751591_alpha-0.003010494989157965_booster-gbtree_max_depth-4_learning_rate-0.006798258260580072_gamma-3.225358230874411e-05_grow_policy-lossguide_2022_10_08_16_17.pkl"

    dump_to_pickle(obj, PATH_TO_DUMP)
