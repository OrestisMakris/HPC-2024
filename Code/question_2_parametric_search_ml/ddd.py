import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# Data generation
X, y = make_classification(
    n_samples=800,
    random_state=42,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    class_sep=0.8
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Parameter grid
params = [{'mlp_layer1': [16, 32], 'mlp_layer2': [16, 32], 'mlp_layer3': [16, 32]}]
pg = list(ParameterGrid(params))  # Convert to a list for easy handling

# Check how many combinations are in the grid
print(f"Parameter grid contains {len(pg)} combinations.")  # Debugging

# Function to train and evaluate a model
def train_and_evaluate(params):
    print(f"Training with params: {params}")  # Debugging
    l1 = params['mlp_layer1']
    l2 = params['mlp_layer2']
    l3 = params['mlp_layer3']
    model = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return params, accuracy

# Use multiprocessing Pool with 6 cores
if __name__ == "__main__":
    # Start time measurement
    start_time = time.time()

    with Pool(processes=8) as pool:  # Set number of processes to 6
        results = pool.map(train_and_evaluate, pg)

    # End time measurement
    end_time = time.time()

    # Display results
    for i, (params, accuracy) in enumerate(results):
        print(f"Model {i}: Params: {params}, Accuracy: {accuracy}")

    # Print total time taken for parallel execution
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
