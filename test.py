import tensorflow as tf

if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
    print("TensorFlow GPU support is available.")
else:
    print("No GPU support detected.")
