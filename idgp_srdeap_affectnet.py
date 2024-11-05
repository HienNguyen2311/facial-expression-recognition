import os, h5py, random, time, operator, joblib
from scoop import futures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import numpy as np
from deap import base, creator, tools, gp, algorithms
import warnings
warnings.filterwarnings('ignore')
import IDGP.evalGP_main as evalGP
from IDGP.evalGP_main import vector_add, vector_sub, vector_mul, vector_div, vector_sin, vector_exp
import IDGP.gp_restrict as gp_restrict
from IDGP.strongGPDataType import Int1, Int2, Int3, Img, Region, Vector, Vector1
import IDGP.feature_function as fe_fs
from IDGP.FER_preprocess_funcs import load_dataset_from_disk, load_checkpoint
from IDGP.FER_fitness_funcs import calculate_sagr, rmse_score

root_fer = '.'
train_dataset_path = os.path.join(root_fer, 'generated_data\\train_images_data_av160.h5')
test_dataset_path = os.path.join(root_fer, 'generated_data\\test_images_data_av80.h5')
chkpoints_dir = os.path.join(root_fer, 'checkpoints\\checkpoints_deapsr')

X_train_val_og, y_train_val_og = load_dataset_from_disk(train_dataset_path)
X_test_og, y_test_og = load_dataset_from_disk(test_dataset_path)

X_train_sq = np.squeeze(X_train_val_og, axis=-1)
x_train = X_train_sq/ 255.0
y_train = y_train_val_og

X_test_sq = np.squeeze(X_test_og, axis=-1)
x_test = X_test_sq/ 255.0
y_test = y_test_og

# parameters:
randomSeeds = 88
population = 30
generation = 5
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8
sagr_alpha = 0.1

bound1, bound2 = x_train[1, :, :].shape

##GP

pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
#Feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector1], Vector1, name='FeaCon')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='FeaCon2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='FeaCon3')


# # Global feature extraction: only select one pset at a time and grey out the rest

# pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
# pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
# pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')

# # Local feature extraction: only select one pset at a time and grey out the rest

# pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
# pset.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
# pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')

# Region detection operators
pset.addPrimitive(fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
pset.addPrimitive(fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
# Terminals
pset.renameArguments(ARG0='Grey')
pset.addEphemeralConstant('X', lambda: random.randint(0, bound1 - 20), Int1)
pset.addEphemeralConstant('Y', lambda: random.randint(0, bound2 - 20), Int2)
pset.addEphemeralConstant('Size', lambda: random.randint(20, 51), Int3)

#fitnesse evaluaiton
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
# toolbox.register("mapp", map)
toolbox.register("mapp", futures.map)

global best_sr_individual
best_sr_individual = None
best_new_features = None
best_fitness = float('inf')
num_features_extracted = 59 #change this number according to the feature extraction type
                            # 32 features per set for Histogram, 20 for DIF, 128 for SIFT, and 59 for uLBP

def evalTrain(individual):
    global best_sr_individual, best_new_features, best_fitness
    print("Individual:", individual)
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    train_tf = np.array(train_tf)
    # print(train_tf)

    num_features = train_tf.shape[1]
    num_feature_sets = num_features // num_features_extracted

    Vector1 = np.ndarray

    pset_sr = gp.PrimitiveSetTyped("MAIN", [Vector1] * num_feature_sets, Vector1)

    # Add these vector operations to the primitive set
    pset_sr.addPrimitive(vector_add, [Vector1, Vector1], Vector1, name='add')
    pset_sr.addPrimitive(vector_sub, [Vector1, Vector1], Vector1, name='sub')
    pset_sr.addPrimitive(vector_mul, [Vector1, Vector1], Vector1, name='mul')
    pset_sr.addPrimitive(vector_div, [Vector1, Vector1], Vector1, name='div')
    pset_sr.addPrimitive(vector_sin, [Vector1], Vector1, name='sin')
    pset_sr.addPrimitive(vector_exp, [Vector1], Vector1, name='exp')

    # Rename arguments to represent each feature set
    for i in range(num_feature_sets):
        pset_sr.renameArguments(**{f'ARG{i}': f'X{i}'})

    # Create a new toolbox for the second stage GP
    toolbox_sr = base.Toolbox()
    toolbox_sr.register("expr", gp.genHalfAndHalf, pset=pset_sr, min_=1, max_=3)
    toolbox_sr.register("individual", tools.initIterate, creator.Individual, toolbox_sr.expr)
    toolbox_sr.register("population", tools.initRepeat, list, toolbox_sr.individual)
    toolbox_sr.register("compile", gp.compile, pset=pset_sr)
    toolbox_sr.register("varAnd", algorithms.varAnd)
    toolbox_sr.register("select", tools.selTournament, tournsize=3)
    toolbox_sr.register("mate", gp.cxOnePoint)
    toolbox_sr.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox_sr.register("mutate", gp.mutUniform, expr=toolbox_sr.expr_mut, pset=pset_sr)
    # toolbox_sr.register("mapp", futures.map, pool)

    # Evolve symbolic regression expressions
    pop_sr = toolbox_sr.population(n=20)
    hof_sr = tools.HallOfFame(1)

    for gen in range(generation):  # Adjust the number of generations as needed
        offspring_sr = toolbox_sr.varAnd(pop_sr, toolbox_sr, cxpb=0.5, mutpb=0.1)
        fits = []
        for ind in offspring_sr:
            func_sr = toolbox_sr.compile(expr=ind)
            feature_sets = [train_tf[:, i*num_features_extracted:(i+1)*num_features_extracted] for i in range(num_feature_sets)]
            feature_sets = np.array(feature_sets)
            # print('feature_sets.shape', feature_sets.shape)
            new_features = func_sr(*feature_sets)
            new_features = np.nan_to_num(new_features, nan=0.0)

            min_max_scaler = preprocessing.MinMaxScaler()
            train_norm = min_max_scaler.fit_transform(np.asarray(new_features))
            # train_norm = new_features
            # print(train_norm.shape)
            # print(np.isnan(train_norm).any())
            svr = SVR()
            multi_output_svr = MultiOutputRegressor(svr)

            # Define custom scorer for RMSE
            rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

            # Perform cross-validation using RMSE as the scoring metric
            scores = cross_val_score(multi_output_svr, train_norm, y_train, cv=3, scoring=rmse_scorer)

            # Calculate mean RMSE (scores are negative RMSE)
            mean_rmse = -scores.mean()

            # For SAGR, you need to fit the model first as SAGR requires actual predictions
            multi_output_svr.fit(train_norm, y_train)
            y_pred = multi_output_svr.predict(train_norm)

            # After training the model
            joblib.dump(multi_output_svr, 'modules\\trained_model.joblib')
            joblib.dump(min_max_scaler, 'modules\\scaler.joblib')

            # Calculate SAGR
            sagr = calculate_sagr(y_train, y_pred)

            # Combine RMSE and SAGR with alpha as the balancing factor
            fitness = mean_rmse + sagr_alpha / sagr
            fits.append((fitness,))

            # Update best individual, features, and fitness
            if fitness < best_fitness:
                best_fitness = fitness
                best_sr_individual = ind
                best_new_features = new_features

    # print('Best symbolic regression individual:', best_sr_individual)
    # print('Best fitness:', best_fitness)

    return best_fitness,

# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint) 
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

# Function to load a checkpoint
def load_checkpoint(filename):
    with open(filename, "rb") as cp_file:
        cp = pickle.load(cp_file)
    return cp

checkpointth = None # Adjust this to run checkpointh (either None or checkpointh number)
start_gen = 1

def GPMain(randomSeeds, start_gen):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()

    if checkpointth is not None:
        checkpoint_filename = os.path.join(chkpoints_dir, f"IDGP_checkpoint_gen_{checkpointth}.pkl") 
        cp = load_checkpoint(checkpoint_filename)
        pop = cp["population"]
        hof = cp["halloffame"]
        log = cp["logbook"]
        random.setstate(cp["rndstate"])
        start_gen = checkpointth

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True, FREQ=1, chkpoints_dir=chkpoints_dir, start_gen=start_gen)

    return pop, log, hof

def evalTest(individual, best_sr_individual):
    # First stage: Extract features using the best individual from the first GP
    func = toolbox.compile(expr=individual)
    test_tf = []
    for i in range(len(y_test)):
        test_tf.append(np.asarray(func(x_test[i, :, :])))
    test_tf = np.array(test_tf)

    # Set up the second stage GP (symbolic regression) primitives
    num_features = test_tf.shape[1]
    num_feature_sets = num_features // num_features_extracted
    pset_sr = gp.PrimitiveSetTyped("MAIN", [Vector1] * num_feature_sets, Vector1)

    # Add vector operations to the primitive set (same as in evalTrain)
    pset_sr.addPrimitive(vector_add, [Vector1, Vector1], Vector1, name='add')
    pset_sr.addPrimitive(vector_sub, [Vector1, Vector1], Vector1, name='sub')
    pset_sr.addPrimitive(vector_mul, [Vector1, Vector1], Vector1, name='mul')
    pset_sr.addPrimitive(vector_div, [Vector1, Vector1], Vector1, name='div')
    pset_sr.addPrimitive(vector_sin, [Vector1], Vector1, name='sin')
    pset_sr.addPrimitive(vector_exp, [Vector1], Vector1, name='exp')

    # Rename arguments for the second stage GP
    for i in range(num_feature_sets):
        pset_sr.renameArguments(**{f'ARG{i}': f'X{i}'})

    # Compile the best symbolic regression individual
    toolbox_sr = base.Toolbox()
    toolbox_sr.register("compile", gp.compile, pset=pset_sr)
    func_sr = toolbox_sr.compile(expr=best_sr_individual)

    # Apply the best symbolic regression expression to the extracted features
    feature_sets = [test_tf[:, i*num_features_extracted:(i+1)*num_features_extracted] for i in range(num_feature_sets)]
    new_features = func_sr(*feature_sets)
    new_features = np.nan_to_num(new_features, nan=0.0)

    # Load the saved scaler and model
    min_max_scaler = joblib.load('modules\\scaler.joblib')
    multi_output_svr = joblib.load('modules\\trained_model.joblib')

   # Transform the test data using the loaded scaler
    test_norm = min_max_scaler.transform(np.asarray(new_features))
    y_pred = multi_output_svr.predict(test_norm)

    # Calculate RMSE and SAGR
    rmse = rmse_score(y_test, y_pred)
    sagr = calculate_sagr(y_test, y_pred)

    # Calculate test fitness
    test_fitness = rmse + sagr_alpha / sagr

    return new_features, test_fitness, rmse, sagr

if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds, start_gen)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    test_features, testResults, rmse, sagr = evalTest(hof[0], best_sr_individual)
    endTime1 = time.process_time()
    testTime = endTime1 - endTime

    print('Number of features', test_features.shape[1])
    print('Best individual ', hof[0])
    print('Best Symbolic Regression Individual', best_sr_individual)
    print('Test loss  ', testResults, 'Test RMSE ', rmse, 'Test SAGR ', sagr)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')
