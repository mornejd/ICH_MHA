from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from Operators import create_ann_model

def evaluate_solution(solution, X_train, y_train, X_test, y_test):
    model = create_ann_model(solution['num_layers'], solution['units_per_layer'], solution['activation_per_layer'], solution['optimizer'], solution['loss'], input_shape=X_train.shape[1])
    model.fit(X_train, y_train, batch_size=solution['batch_size'], epochs=solution['epochs'], verbose=0)
    predictions = model.predict(X_test).round()
    y_pred_proba = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # Precision-Recall curve
    pr, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(rec, pr)
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    return accuracy, precision, recall, f1, sensitivity, specificity, pr_auc, roc_auc
