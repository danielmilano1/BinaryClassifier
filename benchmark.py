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
    print("Benchmark test completed in {:.2f} seconds.".format(execution_time))

# Check if a filename is provided as a command-line argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
    run_benchmark(filename)
else:
    print("Please provide a filename as a command-line argument.")
