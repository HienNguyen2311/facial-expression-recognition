import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_sagr(y_true, y_pred):
    # Ensure y_true and y_pred are numpy arrays for this operation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Initialize counter for correct sign predictions
    sign_agreement_count = 0

    # Loop through each prediction-actual pair
    for true_val, pred_val in zip(y_true, y_pred):
        # For each pair, compare the sign of the actual and predicted value
        sign_agreement_count += np.sum(np.sign(true_val) == np.sign(pred_val))

    # Calculate SAGR by dividing the number of sign agreements by the total number of predictions
    sagr = sign_agreement_count / (2 * len(y_true))  # 2 times the number of instances since we have 2 values per instance
    return sagr