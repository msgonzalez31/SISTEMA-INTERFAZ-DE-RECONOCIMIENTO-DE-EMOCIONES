import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

# --> AÑADIDO: Un diccionario con todos tus modelos
# La clave es el nombre que aparecerá en el menú, el valor es la ruta del archivo.
MODELS = {
    "Seleccionar Modelo": None, # Opción por defecto
    "CNN - 7 Clases": "modelo_entrenado_7clases.h5",
    "CNN - 3 Clases (upsampled)": "modelo_entrenado_upsampled.h5",
    "VGG16 - 3 Clases": "modelo_entrenado_VGG16_3clases.h5",
    "VGG16 - 7 Clases": "modelo_entrenado_VGG16_7clases.h5",
    "ResNet50 - 3 Clases": "modelo_entrenado_ResNet50_3clases_corregido.h5",
    "ResNet50 - 7 Clases": "modelo_ResNet50_7clases_corregido.h5"
}

IMG_SIZE = 100

class EmotionDetectionApp(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.capture = cv2.VideoCapture(0)
        
        # --> MODIFICADO: Atributos para manejar el estado del modelo
        self.model = None
        self.emotion_labels = []
        self.model_path = ""

        self.detection_active = False
        self.update_event = None

    # --> AÑADIDO: La función clave para cargar modelos dinámicamente
    def load_model(self, model_name):
        if self.detection_active:
            self.toggle_detection() # Si la detección está activa, la detenemos

        model_path = MODELS.get(model_name)
        if model_path is None:
            self.model = None
            self.ids.active_model_label.text = "Ningún modelo cargado"
            return
        
        try:
            self.ids.active_model_label.text = f"Cargando: {model_name}..."
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            self.ids.active_model_label.text = f"Modelo Activo: {model_name}"

            # Actualizamos las etiquetas según el modelo cargado
            if "7clases" in self.model_path:
                self.emotion_labels = ['happy', 'sad', 'neutral', 'surprise', 'fear', 'disgust', 'anger']
            else:
                self.emotion_labels = ['Happy', 'Sad', 'Neutral']
            
            print(f"Modelo '{model_name}' cargado. Etiquetas: {self.emotion_labels}")

        except Exception as e:
            self.ids.active_model_label.text = "Error al cargar el modelo"
            print(f"Error: {e}")

    def toggle_detection(self):
        # Verificación para asegurar que un modelo está cargado
        if self.model is None:
            self.ids.emotion_label.text = "Por favor, selecciona un modelo primero"
            return

        self.detection_active = not self.detection_active
        if self.detection_active and self.update_event is None:
            self.ids.welcome_label.opacity = 0
            self.update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)
        elif not self.detection_active and self.update_event is not None:
            self.ids.welcome_label.opacity = 1
            self.update_event.cancel()
            self.update_event = None
            self.ids.emotion_label.text = "Emoción: Detenido"
            self.ids.probability_bar.value = 0

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            
            face_to_predict = face_resized.astype('float32')
            
            # self.model_path para la lógica condicional
            if 'ResNet50' in self.model_path:
                face_processed = resnet50_preprocess(face_to_predict)
            elif 'VGG16' in self.model_path:
                face_processed = face_to_predict / 255.0
            else:
                face_processed = face_to_predict / 255.0

            face_expanded = np.expand_dims(face_processed, axis=0)
            prediction = self.model.predict(face_expanded)[0]
            emotion_index = np.argmax(prediction)
            
            if emotion_index < len(self.emotion_labels):
                emotion_text = self.emotion_labels[emotion_index]
                probability = prediction[emotion_index] * 100
                self.ids.emotion_label.text = f"Emoción: {emotion_text}"
                self.ids.probability_bar.value = probability
            else:
                self.ids.emotion_label.text = "Error: Índice de emoción"
                self.ids.probability_bar.value = 0
        else:
            self.ids.emotion_label.text = "Emoción: No se detecta rostro"
            self.ids.probability_bar.value = 0
        
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.video.texture = texture

    def on_stop(self):
        self.capture.release()

class EmotionApp(App):
    # --> AÑADIDO: Hacemos el diccionario de modelos accesible para el archivo .kv
    MODELS = MODELS 
    def build(self):
        return EmotionDetectionApp()

if __name__ == '__main__':
    EmotionApp().run()