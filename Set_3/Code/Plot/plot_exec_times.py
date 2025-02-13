import matplotlib.pyplot as plt

# Data
N = [512, 1024, 2048, 4096]
gpu_execution_time = [0.066, 0.472, 3.618, 29.675]
cpu_execution_time = [2.779, 26.236, 458.201, 4380.671]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(N, gpu_execution_time, marker='o', linestyle='-', label='GPU Execution Time')
plt.plot(N, cpu_execution_time, marker='s', linestyle='-', label='CPU Execution Time')

# Labels and Title
plt.xlabel('Matrix Dimension (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('GPU vs CPU Execution Time')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Log scale for better visualization
plt.xticks(N)  # Show only the values of N on the x-axis

# Save Plot
plt.savefig('gpu_vs_cpu_execution_time.png')