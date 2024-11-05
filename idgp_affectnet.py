import os, h5py, random, time, operator
from scoop import futures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import numpy as np
from deap import base, creator, tools, gp
import warnings
warnings.filterwarnings('ignore')
import IDGP.evalGP_main as evalGP
import IDGP.gp_restrict as gp_restrict
from IDGP.strongGPDataType import Int1, Int2, Int3, Img, Region, Vector, Vector1
import IDGP.feature_function as fe_fs
from IDGP.FER_preprocess_funcs import load_dataset_from_disk, load_checkpoint
from IDGP.FER_fitness_funcs import calculate_sagr, rmse_score

root_fer = '.'
train_dataset_path = os.path.join(root_fer, 'generated_data\\train_images_data_av160.h5')
test_dataset_path = os.path.join(root_fer, 'generated_data\\test_images_data_av80.h5')
chkpoints_dir = os.path.join(root_fer, 'checkpoints\\checkpoints_svr')

X_train_og, y_train_og = load_dataset_from_disk(train_dataset_path)
X_test_og, y_test_og = load_dataset_from_disk(test_dataset_path)

X_train_sq = np.squeeze(X_train_og, axis=-1)
x_train = X_train_sq/ 255.0
y_train = y_train_og

X_test_sq = np.squeeze(X_test_og, axis=-1)
x_test = X_test_sq/ 255.0
y_test = y_test_og

# parameters:
randomSeeds = 12
population = 30
generation = 15
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
# Global feature extraction
pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
pset.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')
# pset.addPrimitive(fe_fs.all_gabor, [Img], Vector, name='Global_Gabor')
# Local feature extraction
pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
pset.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')
# pset.addPrimitive(fe_fs.all_gabor, [Region], Vector, name='Local_Gabor')
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

def evalTrain(individual):
    print(individual)
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    # print(train_norm.shape)   
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

    # Calculate SAGR
    sagr = calculate_sagr(y_train, y_pred)

    # Combine RMSE and SAGR with alpha as the balancing factor
    fitness = mean_rmse + sagr_alpha / sagr
    return fitness,
 

# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint) 
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

checkpointth = None
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

def evalTest(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    for j in range(0, len(y_test)):
        test_tf.append(np.asarray(func(x_test[j, :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))

    # Fit the SVR model to the training set
    svr = SVR()
    multi_output_svr = MultiOutputRegressor(svr)
    multi_output_svr.fit(train_norm, y_train)

    # Predict the test set
    y_pred = multi_output_svr.predict(test_norm)

    # Calculate RMSE for the test set
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate SAGR for the test set
    sagr = calculate_sagr(y_test, y_pred)

    # Combine RMSE and SAGR with alpha as the balancing factor for the test set fitness
    test_fitness = rmse + sagr_alpha / sagr

    return train_tf, test_tf, test_fitness, rmse, sagr

if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds, start_gen)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    train_features, test_features, testResults, rmse, sagr = evalTest(hof[0])
    endTime1 = time.process_time()
    testTime = endTime1 - endTime

    print('Number of features', train_features.shape[1])
    print('Best individual ', hof[0])
    print('Test loss  ', testResults, 'Test RMSE ', rmse, 'Test SAGR ', sagr)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')
