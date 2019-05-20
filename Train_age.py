import os
import matplotlib.pyplot as plt
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization

# Define
batch_size = 20
target_size = 300
# Build model
input_tensor = Input(shape=(target_size, target_size, 1))
model = keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=input_tensor,
    input_shape=(target_size, target_size, 1),
    pooling='max',
    classes=8)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Chuẩn hóa tập train
train_datagen = ImageDataGenerator(rescale=1. / 255)

# Chuẩn hóa tập validation
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Lấy data
train_generator = train_datagen.flow_from_directory(
    'age',
    target_size=(target_size, target_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    'dataset/validation/age',
    target_size=(target_size, target_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

# Tao thu muc models
if not os.path.exists("models/train/age"):
    os.makedirs("models/train/age")
# Check point
cp_callback = keras.callbacks.ModelCheckpoint("./models/train/age/weights.{epoch:02d}.h5",
                                              save_weights_only=True,
                                              verbose=1)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=31968 // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps= 1600 // batch_size,
    callbacks=[cp_callback])

# Save model
with open('models/train/age/report.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

model_json = model.to_json()
with open("models/train/age/age_model.json", "w") as json_file:
    json_file.write(model_json)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('models/train/age/acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('models/train/age/loss.png')
plt.show()
