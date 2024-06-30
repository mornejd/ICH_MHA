from catboost import CatBoostClassifier

def create_catboost_model(learning_rate, depth, l2_leaf_reg, iterations):
    model = CatBoostClassifier(
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        iterations=iterations,
        verbose=False  # Set to True if you want to see training logs
    )
    return model
