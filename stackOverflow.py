import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import numpy
from sklearn.metrics import confusion_matrix
import seaborn as sns

from keras import layers
from keras import losses
from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer


print(tf.__version__)
# only fetch url once and store it in cache

url = "/stack_overflow_16k.tar.gz"

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

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)
# below we can see the first 2 reviews and labels from the training set.
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(2):
    print(i)
    print("Review", text_batch.numpy()[i])
    print()
    print("Label", label_batch.numpy()[i])
    print()
print()
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])
print()
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'test', 
    batch_size=batch_size
    )

# Next, you will standardize, tokenize, and vectorize the data 
# using the helpful tf.keras.layers.TextVectorization layer.

# def custom_standardization(input_data):
#   lowercase = tf.strings.lower(input_data)
#   stripped_html = tf.strings.regex_replace(lowercase, '', ' ')
#   return tf.strings.regex_replace(stripped_html,
#                                   '[%s]' % re.escape(string.punctuation),
#                                   '')
def custom_standardization(input_data):
    # this will strip HTML break tags '<br />'
    tf.strings.regex_replace(input_data, '<br />', ' ')
    # this will replace punctuation with spaces
    tf.strings.regex_replace(input_data, '[%s]' % re.escape(string.punctuation), '')
    print(input_data)
    return input_data

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

# Tokenization
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(train_text)

# train_sequences = tokenizer.texts_to_sequences(train_text)
# test_sequences = tokenizer.texts_to_sequences(raw_test_ds)

# # Padding
# max_sequence_length = 100  # Define your desired sequence length
# train_pad_seq = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_sequence_length)
# test_pad_seq = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_sequence_length)


# # Normalization
# train_norm = tf.keras.utils.normalize(train_pad_seq)
# test_norm = tf.keras.utils.normalize(test_pad_seq)
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
# print index of strings to integers
print()
print('Strings to ints')
print(vectorize_layer.get_vocabulary()[:10])
print()
# result from using layer
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
# print("Label", raw_train_ds.class_names[first_label])
# print("Review", first_review)
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
  layers.Dense(4),
  layers.Dense(4),
  layers.Activation('softmax')
  ])

model.summary()

# The layers are stacked sequentially to build the classifier:
# The first layer is an Embedding layer. This layer takes the integer-encoded reviews and looks up an embedding vector for each word-index. 
# These vectors are learned as the model trains. The vectors add a dimension to the output array. 
# The resulting dimensions are: (batch, sequence, embedding). To learn more about embeddings, check out the Word embeddings tutorial.
# Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. 
# This allows the model to handle input of variable length, in the simplest way possible.
# The last layer is densely connected with a single output node.

# configure the model to use an optimizer and a loss function:
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Metric to monitor for early stopping
    patience=3,           # Number of epochs with no improvement after which training will be stopped
    verbose=1             # Prints a message when training is stopped
)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=Adam(learning_rate=0.002),
              metrics=['accuracy'])
# You will train the model by passing the dataset object to the fit method.
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[lr_scheduler, early_stopping],
    epochs=epochs)

## Evaluate the model

loss, accuracy = model.evaluate(val_ds)
print()
print("Loss: ", loss)
print("Accuracy: ", accuracy)
print()
def remove_blank(raw_test_ds):
    for text_batch, label_batch in raw_test_ds:
        for i in range(len(text_batch)):
            if text_batch[i] == "blank":
                text_batch[i] = ""
    return raw_test_ds
raw_test_ds = remove_blank(raw_test_ds)
raw_train_ds = remove_blank(raw_train_ds)
raw_val_ds = remove_blank(raw_val_ds)
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
#remove the word "blank" from raw_test_ds

export_model = tf.keras.Sequential([
  vectorize_layer,
  model
])

export_model.compile(
   loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy']
)

model.fit(
   train_ds,
   validation_data=val_ds,
   epochs=50,
   callbacks=[early_stopping]
   )

loss, accuracy = export_model.evaluate(raw_test_ds)
print('Evaluated', accuracy)

# model.predict() for new data
examples = [
  "how to split a string without delimiters? i have string ""513"". i need array [""5"", ""1"", ""3""].",
  "is there a way that i can compare an array and an arraylist? is there a way that i can compare an array and an arraylist like..if (array[0]==arraylist[0])...i am working on this problem",
  "java.lang.string java cannot be cast to java.lang.integer i am getting this error when trying to get data from table.",
  "blankfx serialization textfield i'm using serialization in my blankfx window app. but i can't store user nick (can't serialize blankfx elements) so is there a good option to store it? store it"
]

data = numpy.array(export_model.predict(examples))
print()
print(data)
print()
tf.debugging.set_log_device_placement(True)
# which gpu
print(tf.test.gpu_device_name())
print()
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