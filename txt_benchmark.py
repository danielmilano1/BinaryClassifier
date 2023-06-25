import os
import sys
import tensorflow as tf
import time

def run_benchmark(file_path):
    # Load the file
    data = open(file_path, "r").read()

    # Perform the benchmark test
    start_time = time.time()

    # Your TensorFlow operations on the data file go here
    # Replace this code with your actual benchmarking code

    end_time = time.time()
    execution_time = end_time - start_time

    # Print the benchmark result
    print("Benchmark test for {} completed in {:.6f} seconds.".format(file_path, execution_time))

# Check if a directory is provided as a command-line argument
if len(sys.argv) > 1:
    directory = sys.argv[1]
    if os.path.isdir(directory):
        # Iterate over the files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                run_benchmark(file_path)
    else:
        print("The provided argument is not a directory.")
else:
    print("Please provide a directory as a command-line argument.")
