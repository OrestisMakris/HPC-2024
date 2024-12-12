from sklearn.datasets import make_classification
from multiprocessing import Pool
import os
import csv
import time

def generate_data(seed_chunk):
    seed, n_samples, output_file = seed_chunk

    # Generate synthetic dataset with the specified seed
    X, y = make_classification(
        n_samples=n_samples,
        random_state=seed,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        class_sep=0.8
    )

    # Save data to a CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature1', 'Feature2', 'Label'])  # Write header
        for features, label in zip(X, y):
            writer.writerow([features[0], features[1], label])

    print("Data saved to {}".format(output_file))
    return output_file

def parallel_main():
    # Total samples and threads
    total_samples = 10000
    num_threads = 10
    samples_per_thread = total_samples // num_threads

    # Ensure output directory exists
    output_dir = 'data_chunks'
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for the pool
    args = [(i, samples_per_thread, "{}/chunk_{}.csv".format(output_dir, i)) for i in range(num_threads)]

    # Parallelize data generation
    start_time = time.time()
    with Pool(num_threads) as pool:
        result_files = pool.map(generate_data, args)

    # Combine all chunks into a single CSV
    with open('combined_data.csv', 'w', newline='') as combined_file:
        writer = csv.writer(combined_file)
        writer.writerow(['Feature1', 'Feature2', 'Label'])  # Write header
        for file in result_files:
            with open(file, 'r') as chunk_file:
                reader = csv.reader(chunk_file)
                next(reader)  # Skip header
                writer.writerows(reader)

    end_time = time.time()
    print("All data combined into combined_data.csv")
    print("Parallel execution time: {:.2f} seconds".format(end_time - start_time))

def single_main():
    # Generate entire dataset in a single process
    total_samples = 10000
    output_file = 'combined_data_single.csv'

    start_time = time.time()
    X, y = make_classification(
        n_samples=total_samples,
        random_state=42,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        class_sep=0.8
    )

    # Save data to a CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature1', 'Feature2', 'Label'])  # Write header
        for features, label in zip(X, y):
            writer.writerow([features[0], features[1], label])

    end_time = time.time()
    print("Data saved to {}".format(output_file))
    print("Single-process execution time: {:.2f} seconds".format(end_time - start_time))

if __name__ == '__main__':
    print("Running parallel data generation...")
    parallel_main()
    print("Running single-process data generation for comparison...")
    single_main()
