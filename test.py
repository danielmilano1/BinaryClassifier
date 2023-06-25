import tensorflow as tf

if tf.test.is_built_with_cuda() and tf.test.is_gpu_available():
    print("TensorFlow GPU support is available.")
    if tf.test.is_built_with_tensorrt():
        print("TensorRT integration is successful.")
    else:
        print("TensorRT integration is not found.")
else:
    print("No GPU support detected.")
