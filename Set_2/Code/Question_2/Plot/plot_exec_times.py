import matplotlib.pyplot as plt

# Data
N = [512, 1024, 2048, 4096]
gpu_execution_time = [0.292, 2.111, 16.389, 114.683]
cpu_execution_time = [939, 10839, 155546, 1373761]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(N, gpu_execution_time, marker='o', linestyle='-', label='GPU Execution Time')
plt.plot(N, cpu_execution_time, marker='s', linestyle='-', label='CPU Execution Time')

# Labels and Title
plt.xlabel('Matrix Dimension (N)')
plt.ylabel('Execution Time (ms)')
plt.title('GPU vs CPU Execution Time')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Log scale for better visualization
plt.xticks(N)  # Show only the values of N on the x-axis

# Save Plot
plt.savefig('gpu_vs_cpu_execution_time.png')
