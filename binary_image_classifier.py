#%%
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
from sklearn import metrics


#%%
#model building:

# Note the input shape is the desired size of the image 200x200 with 3 bytes color
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters = 16, 
        kernel_size = (3,3), 
        activation='relu', 
        input_shape=(200, 200, 3)), ##max pooling layer halving the image dimensions: need to change to correspond to the IGV screenshots. 

        # Five convolutional layers: 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 
        tf.keras.layers.Dense(1, activation='sigmoid')])


# %%
model.summary()
# %%
model.compile(loss='binary_crossentropy',
optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
metrics='accuracy')


#%%
#train for 15 epochs
history = model.fit(tf.keras.preprocessing.image.train_generator,
steps_per_epoch=8,
epochs=15,
verbose=1,
validation_data = tf.keras.preprocessing.image.validation_generator,
validation_steps=8)

#%%
model.evaluate(tf.keras.preprocessing.image.validation_generator)

#%%
#ROC curve: 

STEP_SIZE_TEST=tf.keras.preprocessing.image.validation_generator.n//tf.keras.preprocessing.image.validation_generator.batch_size
tf.keras.preprocessing.image.validation_generator.reset()
preds = model.predict(tf.keras.preprocessing.image.validation_generator,
verbose=1)

#%%
fpr, tpr, _ = metrics.roc_curve(tf.keras.preprocessing.image.validation_generator.classes, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()