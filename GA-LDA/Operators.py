from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def create_lda_model(solver, n_components, store_covariance, tol, shrinkage):
    if solver == 'svd':
        shrinkage = None  # Ignore shrinkage for 'svd' solver
    model = LinearDiscriminantAnalysis(
        solver=solver,
        n_components=None if n_components == 0 else n_components,
        store_covariance=store_covariance,
        tol=tol,
        shrinkage=shrinkage
    )
    return model
