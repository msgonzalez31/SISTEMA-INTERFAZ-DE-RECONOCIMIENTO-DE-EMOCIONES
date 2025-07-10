import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample

# Ruta base
dataset_path =  #DEFINIR LA RUTA LOCAL SEGUN SU DISPOTIVO


# Directorios por clase con nombres
image_dirs = {
    'happy': os.path.join(dataset_path, "train", "4"),
    'sad': os.path.join(dataset_path, "train", "5"),
    'neutral': os.path.join(dataset_path, "train", "7"),
    'surprise': os.path.join(dataset_path, "train", "1"),
    'fear': os.path.join(dataset_path, "train", "2"),
    'disgust': os.path.join(dataset_path, "train", "3"),
    'anger': os.path.join(dataset_path, "train", "6"),

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
                img = img.astype('float32') / 255.0
                data.append(img)
                labels.append(i) # Etiqueta numérica (0 a 6)
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
    # Asegurarnos que la clase existe en el set de entrenamiento antes de procesarla
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

# Mezclar los datos para que el modelo no los vea ordenados por clase
shuffle_indices = np.random.permutation(len(X_train_balanced))
X_train_balanced = X_train_balanced[shuffle_indices]
y_train_balanced = y_train_balanced[shuffle_indices]

print(f"\nNuevo tamaño total del set de entrenamiento balanceado: {len(X_train_balanced)} imágenes")

# Convertir etiquetas a formato categórico para Keras
y_train_cat = to_categorical(y_train_balanced, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

# --- Fase 4: Aumento de Datos y Entrenamiento ---
print("\n--- Fase 4: Creando generador y entrenando el modelo ---")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow(X_train_balanced, y_train_cat, batch_size=32)

# Modelo CNN (sin cambios, solo se ajusta la capa final a num_classes)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax') # 7 clases
])

# Compilación (sin cambios)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
history = model.fit(train_generator, epochs=40, validation_data=(X_val, y_val_cat))

# Curvas de aprendizaje (sin cambios)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Precisión del Modelo CNN (7 Clases)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Pérdida del Modelo CNN (7 Clases)')
plt.legend()
plt.tight_layout()
plt.show()

# Matriz de confusión
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
conf_matrix = confusion_matrix(y_val, y_val_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10)) # Hacer la matriz más grande
disp.plot(ax=ax, cmap='viridis', values_format='d')
plt.title("Matriz de Confusión (7 Clases)")
plt.show()

# Reporte de clasificación detallado
print("\n" + "="*50)
print("       Reporte de Clasificación (Métricas de Validación)")
print("="*50)
report = classification_report(y_val, y_val_pred_classes, target_names=class_names)
print(report)
print("="*50 + "\n")

# Guardar el modelo
model.save("modelo_entrenado_7clases.h5")
print("✅ Modelo guardado exitosamente como 'modelo_entrenado_7clases.h5'")