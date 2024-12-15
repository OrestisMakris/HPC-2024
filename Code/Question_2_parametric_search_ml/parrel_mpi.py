from mpi4py import MPI
import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Η συνάρτηση που θα εκτελούν οι Workers
def train_and_evaluate(params):
    # Γεννάμε το dataset
    X, y = make_classification(n_samples=1000, random_state=42, n_features=2,
                               n_informative=2, n_redundant=0, class_sep=0.8)

    # Διαχωρισμός σε training και test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Δημιουργία του μοντέλου MLP
    m = MLPClassifier(hidden_layer_sizes=params, max_iter=2000, solver='adam', random_state=42)

    # Εκπαίδευση του μοντέλου
    m.fit(X_train, y_train)

    # Πρόβλεψη και αξιολόγηση
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_test, y_pred)

    return ac

# Κώδικας για τον Master
def master():
    # Αρχικοποίηση του MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # Ο αριθμός των διεργασιών
    rank = comm.Get_rank()  # Η θέση (rank) της τρέχουσας διεργασίας

    # Ορισμός παραμέτρων για τον Worker
    param_grid = [
        (128, 128, 128, 128, 128),
        (256, 128, 128, 128, 128),
        (128, 256, 128, 128, 128)
    ]

    # Καταγραφή του χρόνου έναρξης
    start_time = time.time()

    # Ο Master στέλνει τα δεδομένα στους Workers
    for i in range(1, size):  # Ο Master είναι η διεργασία με rank 0
        comm.send(param_grid[i-1], dest=i)

    # Ο Master συλλέγει τα αποτελέσματα από τους Workers
    results = []
    for i in range(1, size):
        result = comm.recv(source=i)
        results.append(result)

    # Καταγραφή του χρόνου ολοκλήρωσης
    end_time = time.time()

    # Εκτύπωση των αποτελεσμάτων
    print("Αποτελέσματα από τους Workers:", results)

    # Εκτύπωση του συνολικού χρόνου εκτέλεσης
    total_time = end_time - start_time
    print(f"Συνολικός χρόνος εκτέλεσης: {total_time:.2f} δευτερόλεπτα")

# Κώδικας για τον Worker
def worker():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Η θέση της τρέχουσας διεργασίας

    # Ο Worker περιμένει τα δεδομένα από τον Master
    params = comm.recv(source=0)
    print(f"Worker {rank} επεξεργάζεται τις παραμέτρους: {params}")

    # Εκτέλεση της συνάρτησης train_and_evaluate
    accuracy = train_and_evaluate(params)

    # Ο Worker στέλνει τα αποτελέσματα στον Master
    comm.send(accuracy, dest=0)

# Κύριος κώδικας για να ξεκινήσει η εκτέλεση
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master()
    else:
        worker()
