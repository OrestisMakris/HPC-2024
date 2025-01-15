from mpi4py import MPI
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3, l4, l5), max_iter=2000, solver='adam', random_state=42)

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

pg = list(ParameterGrid(params))

# Master process
if rank == 0:
    start_time = time.time()

    # Divide tasks among workers
    tasks = [[] for _ in range(size)]
    for i, param in enumerate(pg):
        tasks[i % size].append(param)

    # Send tasks to workers
    for i in range(1, size):
        comm.send(tasks[i], dest=i, tag=0)

    # Process tasks assigned to master
    results = [train_and_evaluate(param) for param in tasks[0]]

    # Receive results from workers
    for i in range(1, size):
        results += comm.recv(source=i, tag=1)

    # End timing
    end_time = time.time()

    # Print results
    for r in results:
        print(r)

    # Print total time taken
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

# Worker processes
else:
    # Receive tasks from master
    tasks = comm.recv(source=0, tag=0)

    # Process tasks
    results = [train_and_evaluate(param) for param in tasks]

    # Send results back to master
    comm.send(results, dest=0, tag=1)
