import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

# Configuración de la página
st.set_page_config(page_title="IA Digit Recognizer")
st.title("🔢 Reconocedor de Dígitos en Tiempo Real")
st.write("Dibuja un número del 0 al 9 en el recuadro negro.")

import os
st.write("¿Existe el archivo?:", os.path.exists('modelo_mnist.keras'))
st.write("Archivos en la carpeta:", os.listdir())

# 1. Cargar el modelo guardado
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modelo_mnist.keras')

model = load_my_model()

# 2. Crear el lienzo (Canvas) para dibujar
canvas_result = st_canvas(
    fill_color="white", stroke_width=20, stroke_color="white",
    background_color="black", height=280, width=280, drawing_mode="freedraw", key="canvas",
)

# 3. Procesar el dibujo y predecir
if canvas_result.image_data is not None:
    # Convertir el dibujo a 28x28 píxeles (formato MNIST)
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0  # Normalizar
    
    # Predicción
    pred = model.predict(img.reshape(1, 28, 28, 1))
    clase = np.argmax(pred)
    confianza = np.max(pred)
    
    # 4. Mostrar resultados con Umbral de Seguridad
    st.subheader(f"Resultado: {clase}")
    
    if confianza < 0.80:
        st.warning(f"⚠️ Confianza baja ({confianza:.2%}). ¿Podrías dibujar más claro?")
    else:
        st.success(f"✅ Confianza alta: {confianza:.2%}")
        

    st.bar_chart(pred[0]) # Visualización de probabilidades

