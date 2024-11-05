import operator, math, random, os, h5py, time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error
from IDGP.FER_preprocess_funcs import load_dataset_from_disk, load_checkpoint
from IDGP.FER_fitness_funcs import calculate_sagr, rmse_score

root_fer = '.'
train_dataset_path = os.path.join(root_fer, 'generated_data\\train_features_yy_128px_3kimg.h5')
test_dataset_path = os.path.join(root_fer, 'generated_data\\test_features_yy_128px.h5')

train_features, train_targets = load_dataset_from_disk(train_dataset_path)
test_features, test_targets = load_dataset_from_disk(test_dataset_path)

np.random.seed(9909)
sagr_alpha = 0.1
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_train_features = scaler.fit_transform(train_features)
normalized_test_features = scaler.fit_transform(test_features)

beginTime = time.process_time()
    
pysr_model = PySRRegressor(
    niterations=15,  # Adjust this based on needs
    binary_operators=['+', '*', '-', '/'],
    unary_operators=['cos', 'exp', 'sin', 'log'],
    # loss="loss(x, y) = (x - y)^2",  # MSE loss
    loss='loss(x, y) = (x-y)^2 + 0.1 * (tanh(10*x) == tanh(10*y) ? 1.0 : 0.0)' ,
    ncyclesperiteration=1000,
    maxsize=100, # Adjust this based on needs
    batching=True,
    nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},
    parsimony=0.20,
    bumper=True,
    verbosity=1,
    progress=False,
    procs=100
)

pysr_model.fit(normalized_train_features, train_targets, variable_names=[f"x{i}" for i in range(1536)])
# Get the best equation
best_equation = pysr_model.sympy()

endTime = time.process_time()
trainTime = endTime - beginTime

y_pred = pysr_model.predict(normalized_test_features)
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_targets, y_pred))  
# Calculate SAGR
sagr = calculate_sagr(test_targets, y_pred)   
# Calculate test fitness
test_loss = rmse + sagr_alpha / sagr

print('Best equation ', best_equation)
print('Test loss  ', test_loss, 'Test RMSE ', rmse, 'Test SAGR ', sagr)
print('Train time  ', trainTime)
print('End')