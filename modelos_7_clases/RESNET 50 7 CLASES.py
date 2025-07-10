import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample

# Ruta base
dataset_path =  #DEFINIR LA RUTA LOCAL SEGUN SU DISPOTIVO

image_dirs = {
    'surprise': os.path.join(dataset_path, "train", "1"),
    'fear': os.path.join(dataset_path, "train", "2"),
    'disgust': os.path.join(dataset_path, "train", "3"),
    'happy': os.path.join(dataset_path, "train", "4"),
    'sad': os.path.join(dataset_path, "train", "5"),
    'anger': os.path.join(dataset_path, "train", "6"),
    'neutral': os.path.join(dataset_path, "train", "7"),
}

# Crear una lista ordenada de nombres de clase
class_names = list(image_dirs.keys())
num_classes = len(class_names)

IMG_SIZE = 100
data = []
labels = []

print("--- Fase 1: Carga de TODAS las imágenes originales ---")
for i, (label_name, folder) in enumerate(image_dirs.items()):
    initial_count = len(os.listdir(folder))
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = preprocess_input(img.astype('float32'))
                data.append(img)
                labels.append(i)
    print(f"Clase '{label_name}': {initial_count} imágenes cargadas.")

data = np.array(data)
labels = np.array(labels)

print("\n--- Fase 2: Dividiendo en sets de Entrenamiento y Validación (85/15) ---")
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.15, random_state=42, stratify=labels)

print(f"Set de Entrenamiento Original: {len(X_train)} imágenes")
print(f"Set de Validación: {len(X_val)} imágenes (¡Este set no se tocará!)")

print("\n--- Fase 3: Balanceando el set de entrenamiento por sobremuestreo ---")
unique, counts = np.unique(y_train, return_counts=True)
max_samples = np.max(counts)
print(f"La clase mayoritaria tiene {max_samples} muestras. Se sobremuestrearán las demás clases para igualarla.")

X_train_balanced_list = []
y_train_balanced_list = []

for class_index in range(num_classes):
    if class_index in unique:
        X_class = X_train[y_train == class_index]
        y_class = y_train[y_train == class_index]
        X_resampled, y_resampled = resample(X_class, y_class,
                                            replace=True,
                                            n_samples=max_samples,
                                            random_state=42)
        X_train_balanced_list.append(X_resampled)
        y_train_balanced_list.append(y_resampled)
        print(f"Clase '{class_names[class_index]}' ahora tiene {len(X_resampled)} muestras en el set de entrenamiento.")

X_train_balanced = np.concatenate(X_train_balanced_list)
y_train_balanced = np.concatenate(y_train_balanced_list)

shuffle_indices = np.random.permutation(len(X_train_balanced))
X_train_balanced = X_train_balanced[shuffle_indices]
y_train_balanced = y_train_balanced[shuffle_indices]

print(f"\nNuevo tamaño total del set de entrenamiento balanceado: {len(X_train_balanced)} imágenes")

y_train_cat = to_categorical(y_train_balanced, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)


print("\n--- Fase 4: Creando generador y entrenando el modelo ResNet50 ---")
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow(X_train_balanced, y_train_cat, batch_size=32)

# Cargar ResNet50 sin top
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

# Modelo final
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

# Compilación
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
optimizer = Adam(learning_rate=1e-4) # 1e-4 es 0.0001
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Entrenar
history = model.fit(train_generator, epochs=40, validation_data=(X_val, y_val_cat))

# Curvas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Accuracy Validación')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Precisión - ResNet50 (7 Clases)')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss Validación')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Pérdida - ResNet50 (7 Clases)')
plt.tight_layout()
plt.show()

# Matriz de confusión
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
conf_matrix = confusion_matrix(y_val, y_val_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='viridis', values_format='d')
plt.title("Matriz de Confusión (ResNet50 - 7 Clases)")
plt.show()

# Reporte de clasificación detallado
print("\n" + "="*50)
print("       Reporte de Clasificación (Métricas de Validación)")
print("="*50)
report = classification_report(y_val, y_val_pred_classes, target_names=class_names)
print(report)
print("="*50 + "\n")

# Guardar el modelo
model.save("modelo_resnet50_7clases_corregido.h5")
print("✅ Modelo guardado exitosamente como 'modelo_resnet50_7clases_corregido.h5'")