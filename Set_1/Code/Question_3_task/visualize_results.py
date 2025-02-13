#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read CSV file
    df = pd.read_csv("results.csv")

    # Ensure numeric columns are correctly typed
    df['ArraySize'] = pd.to_numeric(df['ArraySize'], errors='coerce')
    df['NumThreads'] = pd.to_numeric(df['NumThreads'], errors='coerce')
    df['ExecutionTime'] = pd.to_numeric(df['ExecutionTime'], errors='coerce')

    # ----------------------------
    # Plot 1: Execution Time vs. Number of Threads (Separate plot per Array Size)
    # ----------------------------
    for array_size in sorted(df['ArraySize'].unique()):
        plt.figure(figsize=(10, 6))
        df_subset = df[df['ArraySize'] == array_size]

        for method in df_subset['Method'].unique():
            df_method = df_subset[df_subset['Method'] == method]
            plt.plot(df_method['NumThreads'], df_method['ExecutionTime'], marker='o', linestyle='-', label=method)

        plt.xlabel("Number of Threads")
        plt.ylabel("Execution Time (seconds)")
        plt.title(f"Execution Time vs. Number of Threads (ArraySize={int(array_size)})")
        plt.legend()
        plt.grid(True)
        plt.xscale('log', base=2)  # If thread counts are powers of 2
        plt.savefig(f"execution_time_threads_{int(array_size)}.png")
        plt.show()

    # ----------------------------
    # Plot 2: Execution Time vs. Array Size (Separate plot per Thread Count)
    # ----------------------------
    for nthreads in sorted(df['NumThreads'].unique()):
        plt.figure(figsize=(10, 6))
        df_subset = df[df['NumThreads'] == nthreads]

        for method in df_subset['Method'].unique():
            df_method = df_subset[df_subset['Method'] == method]
            df_method = df_method.sort_values(by='ArraySize')
            plt.plot(df_method['ArraySize'], df_method['ExecutionTime'], marker='o', linestyle='-', label=method)

        plt.xlabel("Array Size")
        plt.ylabel("Execution Time (seconds)")
        plt.title(f"Execution Time vs. Array Size (Threads={int(nthreads)})")
        plt.legend()
        plt.grid(True)
        plt.xscale('log')  # Use log scale if array sizes vary widely
        plt.savefig(f"execution_time_array_{int(nthreads)}.png")
        plt.show()

if __name__ == "__main__":
    main()
