import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class_df = pd.read_csv('traffic_sign.csv')
class_names = class_df['Name'].tolist()
num_classes = len(class_names)

IMG_SIZE = (32, 32)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    'Images/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    'Images/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Adaptive Gradient Control (AGC) optimizer wrapper
class AGC(tf.keras.optimizers.Adam):
    def _resource_apply_dense(self, grad, var, apply_state):
        grad_norm = tf.linalg.global_norm([grad])
        agc_factor = tf.minimum(self.learning_rate / (grad_norm + 1e-7), 1.0)
        grad = grad * agc_factor
        return super()._resource_apply_dense(grad, var, apply_state)

# Improved CNN Architecture with hierarchical structure
def create_cnn_model():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Classification head
        layers.Flatten(),
        layers.Dense(256, kernel_initializer='he_normal'),
        layers.LeakyReLU(alpha=0.3),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Custom AGC optimizer with initial learning rate 0.001
    optimizer = AGC(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_cnn_model()

history = model.fit(
    train_generator,
    epochs=350,
    validation_data=val_generator
)

model.save('traffic_sign_cnn_32x32__1.h5')

# Adaptive Reward Function with confidence scaling
def adaptive_reward_function(true_label, predicted_probs):
    predicted_label = np.argmax(predicted_probs)
    true_label_prob = predicted_probs[true_label]
    
    if predicted_label == true_label:
        # Scale reward by confidence level
        reward = np.clip(true_label_prob * 2, 0.5, 2.0)
    else:
        # Penalize based on confidence in wrong prediction and margin
        max_incorrect = np.max(predicted_probs[np.arange(num_classes) != true_label])
        reward = -np.clip((max_incorrect - true_label_prob) * 2, 0.5, 2.0)
    
    return reward

# Evaluation metrics
val_pred = model.predict(val_generator)
val_pred_classes = np.argmax(val_pred, axis=1)
val_true_classes = val_generator.classes

# Generate classification report
report = classification_report(val_true_classes, val_pred_classes, 
                              target_names=class_names, output_dict=True)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

conf_matrix = confusion_matrix(val_true_classes, val_pred_classes)
specificity = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1_score}")

# Example of adaptive reward usage
sample_image, sample_label = val_generator[0][0][0], val_generator[0][1][0]
sample_pred = model.predict(np.expand_dims(sample_image, axis=0))[0]
true_label_idx = np.argmax(sample_label)
reward = adaptive_reward_function(true_label_idx, sample_pred)
print(f"Adaptive Reward: {reward:.2f}")

# Visualizations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()