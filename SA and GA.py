#Simulated Annealing Algorithm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import shap
import copy
import math
import random

# Set all columns except the one with the header name "TG" as x, and set the column with the header name "TG" as y.
df = pd.read_excel('model.xlsx')
TG_column = df['TG']
df = df.drop('TG', axis=1)

y = TG_column
x = df.values
X = np.array(x)
y = np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=48)

# Data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
mlp = MLPRegressor(random_state=48, batch_size=1)

# Define the search space for hyperparameters
param_space = {
    'hidden_layer_sizes': [(23,), (48,), (48, 48), (96, 96)],
    'activation': ['relu', 'tanh'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

# Define the parameters for the simulated annealing algorithm
T_max = 100  
T_min = 0.1  
iter_max = 100  
delta = 0.95  
scale = 0.1  

# Initialize the best parameters and their score
best_params = None
best_score = float('inf')

# Simulated annealing algorithm
params = {k: random.choice(v) for k, v in param_space.items()}  # 随机初始化参数
T = T_max
while T > T_min:
    for i in range(iter_max):
        params_new = copy.deepcopy(params)
        param_name = random.choice(list(param_space.keys()))
        param_value = random.choice(param_space[param_name])
        params_new[param_name] = param_value
        mlp_new = MLPRegressor(**params_new, random_state=42, batch_size=1)
        mlp_new.fit(X_train_scaled, y_train)
        y_pred = mlp_new.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        delta_score = mse - best_score
        if delta_score < 0:
            params = params_new
            best_score = mse
            if best_score < mse:
                best_mlp = mlp_new
        else:
            prob = math.exp(-delta_score/T)
            if random.random() < prob:
                params = params_new
        T *= delta

# Validate the model in the test dataset
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE: {:.4f}'.format(mse))
print('MAE: {:.4f}'.format(mae))
print('R^2: {:.4f}'.format(r2))

# SHAP calculation and visualization
feature_names = df.columns
explainer = shap.Explainer(mlp_new.predict, X_test_scaled)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)

##################Split###################################

#Genetic Algorithm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import random

# Read data and preprocess
df = pd.read_excel('model.xlsx')
TG_column = df['TG']
df = df.drop('TG', axis=1)

y = TG_column
x = df.values
X = np.array(x)
y = np.array(y)

# Generate initial population
def init_population(pop_size, chromosome_size):
    population = []
    for i in range(pop_size):
        chromosome = [random.uniform(-1, 1) for j in range(chromosome_size)]
        population.append(chromosome)
    return population

# Fitness function for individuals
def fitness_function(chromosome):
    mlp = MLPRegressor(hidden_layer_sizes=(18,12), activation='relu', solver='adam', max_iter=100)
    mlp.fit(X_train * chromosome, y_train)
    y_pred = mlp.predict(X_val * chromosome)
    mse = mean_squared_error(y_val, y_pred)
    return 1 / (mse + 1e-6)

# Selection
def selection(population, fitness_values):
    fitness_sum = np.sum(fitness_values)
    probs = fitness_values / fitness_sum
    probs_cum = np.cumsum(probs)
    selected = []
    for i in range(len(population)):
        rand = np.random.rand()
        for j in range(len(population)):
            if rand <= probs_cum[j]:
                selected.append(population[j])
                break
    return selected

# Crossover
def crossover(population, crossover_rate):
    for i in range(len(population)):
        if np.random.rand() < crossover_rate:
            j = np.random.randint(len(population))
            while j == i:
                j = np.random.randint(len(population))
            parent1 = population[i]
            parent2 = population[j]
            k = np.random.randint(len(parent1))
            child1 = parent1[:k] + parent2[k:]
            child2 = parent2[:k] + parent1[k:]
            population[i] = child1
            population[j] = child2

# Mutation
def mutation(population, mutation_rate):
    for i in range(len(population)):
        if np.random.rand() < mutation_rate:
            chromosome = population[i]
            j = np.random.randint(len(chromosome))
            chromosome[j] = random.uniform(-1, 1)

# Calculate model performance metrics
def calculate_metrics(chromosome):
    mlp = MLPRegressor(hidden_layer_sizes=(18,12), activation='relu', solver='adam', max_iter=100)
    mlp.fit(X_train * chromosome, y_train)
    y_pred = mlp.predict(X_test * chromosome)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

# Cross-validation
def cross_validate(chromosome):
    mlp = MLPRegressor(hidden_layer_sizes=(18,12), activation='relu', solver='adam', max_iter=100)
    mlp.fit(X_train * chromosome, y_train)
    scores = cross_val_score(mlp, X_train * chromosome, y_train, cv=5, scoring='neg_mean_squared_error')
    fitness = np.mean(scores)
    return fitness

# Feature importance and SHAP values
def explain_model(chromosome):
    mlp = MLPRegressor(hidden_layer_sizes=(18,12), activation='relu', solver='adam', max_iter=1000)
    mlp.fit(X_train * chromosome, y_train)
    explainer = shap.KernelExplainer(mlp.predict, X_train * chromosome)
    shap_values = explainer.shap_values(X_test * chromosome)
    shap.summary_plot(shap_values, X_test * chromosome, plot_type="bar")
    return shap_values

# Split dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Initialize parameters
POP_SIZE = 20
CHROMO_SIZE = X_train.shape[1]
MAX_GENERATION = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# Generate initial population
population = init_population(POP_SIZE, CHROMO_SIZE)

# Genetic algorithm optimization
for i in range(MAX_GENERATION):
    fitness_values = [fitness_function(chromosome) for chromosome in population]
    fittest_index = np.argmax(fitness_values)
    fittest_chromosome = population[fittest_index]
    mae, mse, rmse, r2 = calculate_metrics(fittest_chromosome)
    print(f'Generation {i+1} - Best RMSE: {rmse:.4f} - R^2: {r2:.4f} - MAE: {mae:.4f} - MSE: {mse:.4f}')
    if i == 0:
        shap_values = explain_model(fittest_chromosome)
        print('Feature importance based on SHAP values for the fittest chromosome:')
    population = selection(population, fitness_values)
    crossover(population, CROSSOVER_RATE)
    mutation(population, MUTATION_RATE)
    
# Cross-validation
fitness_values = [cross_validate(chromosome) for chromosome in population]
fittest_index = np.argmax(fitness_values)
fittest_chromosome = population[fittest_index]
mae, mse, rmse, r2 = calculate_metrics(fittest_chromosome)
print(f'Best fitness value: {fitness_values[fittest_index]:.4f} - Best RMSE: {rmse:.4f} - R^2: {r2:.4f} - MAE: {mae:.4f} - MSE: {mse:.4f}')
shap_values = explain_model(fittest_chromosome)
print('Feature importance based on SHAP values for the fittest chromosome:')

# SHAP calculation and visualization
feature_names = df.columns
explainer = shap.Explainer(mlp_new.predict, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)