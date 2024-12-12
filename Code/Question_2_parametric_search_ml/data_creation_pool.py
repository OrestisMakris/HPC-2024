from sklearn.datasets import make_classification
from multiprocessing import Pool
import csv
import time

# Constants for dataset generation
N_FEATURES = 18
N_INFORMATIVE = 2
N_REDUNDANT = 0
CLASS_SEP = 0.8
TOTAL_SAMPLES = 100000
NUM_THREADS = 20


def generate_data_in_memory(seed_chunk):
    seed, n_samples = seed_chunk
    # Generate synthetic dataset with the specified seed
    X, y = make_classification(
        n_samples=n_samples,
        random_state=seed,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        class_sep=CLASS_SEP
    )
    # Return the generated data as a list of rows
    return [[features[0], features[1], label] for features, label in zip(X, y)]


def write_to_csv(file_path, data):
    # Write the generated data to a CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature1', 'Feature2', 'Label'])  # Write header
        writer.writerows(data)


def parallel_main():
    samples_per_thread = TOTAL_SAMPLES // NUM_THREADS
    args = [(i, samples_per_thread) for i in range(NUM_THREADS)]

    # Track time for data generation, aggregation, and writing
    start_time = time.time()

    paralle_gen_start_time = time.time()
    # Parallelize data generation
    with Pool(NUM_THREADS) as pool:
        all_data = pool.map(generate_data_in_memory, args)
    paralle_gen_end_time = time.time()

    # Aggregate data (combine chunks into a single list)
    aggregate_start_time = time.time()
    combined_data = [row for chunk in all_data for row in chunk]
    aggregate_end_time = time.time()

    # Write the combined data to a CSV file
    output_file = 'combined_data.csv'
    write_start_time = time.time()
    write_to_csv(output_file, combined_data)
    write_end_time = time.time()

    end_time = time.time()

    print("Parallel execution completed.")
    print(f"Data generation time: {paralle_gen_end_time -paralle_gen_start_time =:.2f} seconds")
    print(f"Data  aggregation time: {aggregate_end_time - aggregate_start_time:.2f} seconds")
    print(f"CSV writing time: {write_end_time - write_start_time:.2f} seconds")
    print(f"Total parallel execution time: {end_time - start_time:.2f} seconds")


def single_main():
    samples_per_thread = TOTAL_SAMPLES // NUM_THREADS
    args = [(i, samples_per_thread) for i in range(NUM_THREADS)]

    # Track time for data generation, aggregation, and writing
    start_time = time.time()

    # Generate synthetic data using the same logic as the parallel version
    paralle_gen_start_time = time.time()
    all_data = [generate_data_in_memory(chunk) for chunk in args]  # Generate data for each chunk
    paralle_gen_end_time = time.time()

    # Aggregate data (combine chunks into a single list)
    aggregate_start_time = time.time()
    combined_data = [row for chunk in all_data for row in chunk]
    aggregate_end_time = time.time()

    # Write the combined data to a CSV file
    output_file = 'combined_data_single.csv'
    write_start_time = time.time()
    write_to_csv(output_file, combined_data)
    write_end_time = time.time()

    end_time = time.time()

    print("Single-process execution completed.")
    print(f"Data generation time: {paralle_gen_end_time - paralle_gen_start_time:.2f} seconds")
    print(f"Data aggregation time: {aggregate_end_time - aggregate_start_time:.2f} seconds")
    print(f"CSV writing time: {write_end_time - write_start_time:.2f} seconds")
    print(f"Total single-process execution time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    print("Running parallel data generation...")
    parallel_main()
    print("---------------------------------------------------------")
    print("Running single-process data generation for comparison...")
    single_main()
