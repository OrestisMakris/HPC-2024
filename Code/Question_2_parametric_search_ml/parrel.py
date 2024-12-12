from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP to 1 thread
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL to 1 thread

# Function to generate data and train the model
def train_and_evaluate(params):
    # Generate dataset inside the worker
    X, y = make_classification(n_samples=1000, random_state=42, n_features=2,
                               n_informative=2, n_redundant=0, class_sep=0.8)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Extract parameters
    l1 = params['mlp_layer1']
    l2 = params['mlp_layer2']
    l3 = params['mlp_layer3']
    l4 = params['mlp_layer4']  # New layer
    l5 = params['mlp_layer5']  # New layer

    # Define MLP classifier with more hidden layers, more neurons, and larger epochs
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3, l4, l5), max_iter=1000, solver='adam', random_state=42)

    # Fit the classifier
    m.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_test, y_pred)

    return (params, ac)


# Define parameter grid for larger networks
params = [{'mlp_layer1': [128, 256],
           'mlp_layer2': [128, 256],
           'mlp_layer3': [128, 256],
           'mlp_layer4': [128, 256],  # New layer choices
           'mlp_layer5': [128, 256]}]  # New layer choices

pg = ParameterGrid(params)

# Start timing
start_time = time.time()

# Create a Pool of workers (e.g., using 4 workers)
with Pool(processes=14) as pool:
    results = pool.map(train_and_evaluate, pg)

# End timing
end_time = time.time()

# Print results
for r in results:
    print(r)

# Print total time taken
print(f"Total time taken: {end_time - start_time:.2f} seconds")
