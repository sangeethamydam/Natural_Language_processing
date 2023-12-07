import timeit
import pandas as pd
import matplotlib.pyplot as plt

# Simulate Polynomial Ring Learning With Errors (PRLWE) Encryption
def simulate_prlwe_encryption():
    start_time = timeit.default_timer()
    # Replace this with your specific PRLWE encryption logic
    elapsed_time = timeit.default_timer() - start_time
    iterations = 200  # Replace with the actual iterations count
    return elapsed_time, iterations

# Simulate Polynomial Ring Learning With Errors (PRLWE) Decryption
def simulate_prlwe_decryption():
    start_time = timeit.default_timer()
    # Replace this with your specific PRLWE decryption logic
    elapsed_time = timeit.default_timer() - start_time
    iterations = 100  # Replace with the actual iterations count
    return elapsed_time, iterations

# Simulate Proposed Approach Encryption
def simulate_proposed_encryption():
    start_time = timeit.default_timer()
    # Replace this with your specific proposed encryption logic
    elapsed_time = timeit.default_timer() - start_time
    iterations = 150  # Replace with the actual iterations count
    return elapsed_time, iterations

# Simulate Proposed Approach Decryption
def simulate_proposed_decryption():
    start_time = timeit.default_timer()
    # Replace this with your specific proposed decryption logic
    elapsed_time = timeit.default_timer() - start_time
    iterations = 80  # Replace with the actual iterations count
    return elapsed_time, iterations

# Simulate multiple iterations for statistical analysis
num_iterations = 10

# Algorithms to compare
algorithms = ['PRLWE Encryption', 'PRLWE Decryption', 'Proposed Encryption', 'Proposed Decryption']
data = {f'{algo} Time': [] for algo in algorithms}
data.update({f'{algo} Iterations': [] for algo in algorithms})

for _ in range(num_iterations):
    for algo in algorithms:
        if 'PRLWE' in algo:
            if 'Encryption' in algo:
                algo_time, algo_iterations = simulate_prlwe_encryption()
            else:
                algo_time, algo_iterations = simulate_prlwe_decryption()
        else:
            if 'Encryption' in algo:
                algo_time, algo_iterations = simulate_proposed_encryption()
            else:
                algo_time, algo_iterations = simulate_proposed_decryption()

        data[f'{algo} Time'].append(algo_time)
        data[f'{algo} Iterations'].append(algo_iterations)

# Create a DataFrame for tabular data
df = pd.DataFrame(data)

# Calculate average values
averages = df.mean()

# Save data to Excel file
excel_filename = 'algorithm_comparison.xlsx'
df.to_excel(excel_filename, index=False)

# Print the table
print("Performance Metrics:")
print(df)
print("\nAverage Values:")
print(averages)

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=len(algorithms), figsize=(15, 8))

for i, algo in enumerate(algorithms):
    # Plot Time
    axes[0, i].bar(df.index, df[f'{algo} Time'], color='blue')
    axes[0, i].set_title(f'{algo} Time')
    axes[0, i].set_ylabel('Time (s)')

    # Plot Iterations
    axes[1, i].bar(df.index, df[f'{algo} Iterations'], color='orange')
    axes[1, i].set_title(f'{algo} Iterations')
    axes[1, i].set_ylabel('Iterations')

plt.tight_layout()
plt.savefig('algorithm_comparison_graphs.png')
plt.show()
