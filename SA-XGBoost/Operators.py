from xgboost import XGBClassifier

def create_xgboost_model(C, dual, loss, penalty, tol, learning_rate, max_depth, max_features, min_samples_leaf, min_samples_split, n_estimators, subsample):
    model = XGBClassifier(
        C=C,
        dual=dual,
        loss=loss,
        penalty=penalty,
        tol=tol,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        subsample=subsample,
        use_label_encoder=False,  # Set to False to avoid unnecessary warnings
        eval_metric='logloss'  # Default evaluation metric for XGBoost
    )
    return model
