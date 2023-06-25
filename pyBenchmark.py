import os
import sys
import time

def run_benchmark(file_path):
    # Perform the benchmark test
    start_time = time.time()

    # Execute the Python script
    exec(open(file_path).read())

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
            if os.path.isfile(file_path) and file_path.endswith(".py"):
                run_benchmark(file_path)
    else:
        print("The provided argument is not a directory.")
else:
    print("Please provide a directory as a command-line argument.")
