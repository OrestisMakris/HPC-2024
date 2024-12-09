from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

# Create dataset
X, y = make_classification(n_samples=10000, random_state=42, n_features=2,
                           n_informative=2, n_redundant=0, class_sep=0.8)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define parameter grid
params = [{'mlp_layer1': [16, 32],
           'mlp_layer2': [16, 32],
           'mlp_layer3': [16, 32]}]

pg = ParameterGrid(params)

# Store results
results = []

# Start timing
start_time = time.time()

# Iterate over parameter combinations
for i, p in enumerate(pg):
    print(p)
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']

    # Define MLP classifier with specified layer sizes
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))

    # Fit the classifier
    m.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)
    print(i, ac)

    # Append results
    results.append((i, p, ac))

# End timing
end_time = time.time()

# Print results
for r in results:
    print(r)

# Print total time taken
print(f"Total time taken: {end_time - start_time:.2f} seconds")
