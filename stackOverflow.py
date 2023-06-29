import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import numpy

from keras import layers
from keras import losses
from keras.optimizers import Adam


print(tf.__version__)

url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

train_dir = os.path.join(os.path.dirname(dataset), 'train')
os.listdir(train_dir)

python_dir = os.path.join(train_dir, 'python')
os.listdir(python_dir)

js_dir = os.path.join(train_dir, 'javascript')
os.listdir(js_dir)

csharp_dir = os.path.join(train_dir, 'csharp')
os.listdir(csharp_dir)

java_dir = os.path.join(train_dir, 'java')
os.listdir(java_dir)

# sample_file = os.path.join(python_dir, '1.txt')
# with open(sample_file) as f:
#   print(f.read())

# remove_dir = os.path.join(train_dir, 'unsup')
# shutil.rmtree(remove_dir)

# When running a machine learning experiment,
# it is a best practice to divide your dataset 
# into three splits: train, validation, and test.

batch_size = 36
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

# for text_batch, label_batch in raw_train_ds.take(1):
#   for i in range(2):
#     print(i)
#     print("Review", text_batch.numpy()[i])
#     print("Label", label_batch.numpy()[i])
#     print()

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'train', 
    batch_size=batch_size)

# Next, you will standardize, tokenize, and vectorize the data 
# using the helpful tf.keras.layers.TextVectorization layer.

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

# Next, you will create a TextVectorization layer. You will use this layer to standardize, 
# tokenize, and vectorize our data. You set the output_mode to int 
# to create unique integer indices for each token.

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Next, you will call adapt to fit the state of the preprocessing layer to the dataset. This will cause the model to build an index of strings to integers.

# Note: It's important to only use your training data when calling adapt (using the test set would leak information).

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# result from using layer
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
# if the review contains the word blank, remove it, then insert the first_label in its place
first_review = tf.where(first_review == b'blank', first_label, first_review)
print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))

vocabulary = vectorize_layer.get_vocabulary()
if len(vocabulary) > 1287:
    print("1287 ---> ", vocabulary[1287])
else:
    print("Index is out of range.")

print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# As a final preprocessing step, you will apply the TextVectorization layer 
# you created earlier to the train, validation, and test dataset.

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Enhance performance

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#### Create the model ######
embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.1),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.1),
  layers.Dense(4)])

model.summary()

# The layers are stacked sequentially to build the classifier:
# The first layer is an Embedding layer. This layer takes the integer-encoded reviews and looks up an embedding vector for each word-index. 
# These vectors are learned as the model trains. The vectors add a dimension to the output array. 
# The resulting dimensions are: (batch, sequence, embedding). To learn more about embeddings, check out the Word embeddings tutorial.
# Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. 
# This allows the model to handle input of variable length, in the simplest way possible.
# The last layer is densely connected with a single output node.

# configure the model to use an optimizer and a loss function:
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=Adam(learning_rate=0.005),
              metrics=['accuracy'])
# You will train the model by passing the dataset object to the fit method.
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[lr_scheduler],
    epochs=epochs)

## Evaluate the model

loss, accuracy = model.evaluate(val_ds)
print()
print("Loss: ", loss)
print("Accuracy: ", accuracy)
print()
# model.fit() returns a History object that 
# contains a dictionary with everything that happened during training:

history_dict = history.history
history_dict.keys()

# 4 entries, 1 for each monitored metric
# You can use these to plot the training and validation loss for comparison, 
# as well as the training and validation accuracy:
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# lrs = 1e-4 * (10 ** (numpy.arange(100) / 20))
# plt.figure(figsize=(10, 7))
# plt.plot(lrs, loss, 'bo', label='Learning loss')
# plt.semilogx(lrs, history_3.history["loss"] )
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Learning rate vs Loss")

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# In this plot, the dots represent the training loss and accuracy, and the solid lines are the validation loss and accuracy.

# Notice the training loss decreases with each epoch and the training accuracy increases with each epoch. 
# This is expected when using a gradient descent optimization
# —it should minimize the desired quantity on every iteration.

# For this particular case, you could prevent overfitting by simply stopping the training when the validation accuracy is no longer increasing.
# One way to do so is to use the tf.keras.callbacks.EarlyStopping callback.

# you can include the TextVectorization layer inside your model. To do so, you can create a new model using the weights you just trained.
# go through raw_test_ds, remove the word "blank" and return new raw_test_ds
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)
model.fit(train_ds)
# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print('Evaluated', accuracy)

# model.predict() for new data
examples = [
  "incrementing a for loop by decimal value i'm trying to implement a for loop that increments by 0.1. i have one that seems to work just fine for an Javascript?",
  "callback function in javascript",
  "calculating mileage rates i'm creating a hr mileage and expenses system but am struggling to come up with ",
]

data = numpy.array(export_model.predict(examples))

max_indices = numpy.argmax(data, axis=1)

types = ['python', 'javascript', 'csharp', 'java']

for i in range(len(max_indices)):
    print(max_indices[i])
    print(examples[i])
    print(types[max_indices[i]])
    print()



# There is a performance difference to keep in mind when choosing where to apply your TextVectorization layer. Using it outside of your 
# model enables you to do asynchronous CPU processing and buffering of your data when training on GPU. So, if you're training your model
#  on the GPU, you probably want to go with this option to get the best performance while developing your model, then switch to including
#  the TextVectorization layer inside your model when you're ready to prepare for deployment.