
#Importamos las librerias necesarias para el desarrollo del clasificador

import glob
import numpy as np
from keras import models, layers, utils
from sklearn.model_selection import train_test_split 
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
"""
 Cargamos la imagenes de peras y manzanas. Estas se encuentran dentro de sus respectivas carpetas. 
"""
peras = glob.glob("Peras/*.jpg")
manzanas = glob.glob("Manzanas/*.jpg")

imagenes = peras + manzanas
etiquetas = [0] * len(peras) + [1] * len(manzanas)

# Ahora, procesamos las imágenes y las convertimos en un formato que pueda ser utilizado por la red neuronal. En este caso, vamos a redimensionar las imágenes a 32x32 pixels y convertirlas a escala de grises.

def procesar_imagen(address):
    img_width = 32
    img_height = 32
    rgb = io.imread(address)
    rgb_resized = resize(rgb, (img_height, img_width), anti_aliasing=True)
    gray_resized = img_as_ubyte(rgb2gray(rgb_resized))
    return gray_resized

imagenes_procesadas = [procesar_imagen(img) for img in imagenes]

# Luego, se  dividen las imágenes en conjuntos de entrenamiento y prueba.

X = np.array(imagenes_procesadas)
y = np.array(etiquetas)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #42

# Ahora, podemos crear el modelo de red neuronal convolucional.

from keras.layers import Dropout

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(layers.Dense(2, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Finalmente, podemos entrenar el modelo con los conjuntos de entrenamiento y prueba.

hist = model.fit(X_train, utils.to_categorical(y_train), validation_data=(X_test, utils.to_categorical(y_test)), epochs=300, verbose=False)

loss, accuracy = model.evaluate(X_test, utils.to_categorical(y_test))
print(f'Precisión final del modelo: {accuracy:.2f}')

def probar_modelo(ruta_imagen):
    # Procesar la imagen de la misma manera que se procesaron las imágenes de entrenamiento
    img_width = 32
    img_height = 32
    rgb = io.imread(ruta_imagen)
    rgb_resized = resize(rgb, (img_height, img_width), anti_aliasing=True)
    gray_resized = img_as_ubyte(rgb2gray(rgb_resized))
    img_procesada = gray_resized.reshape((1, img_height, img_width, 1))

    # Realizar la predicción con el modelo
    prediccion = model.predict(img_procesada)

    # Convertir la predicción en una etiqueta (pera o manzana)
    etiqueta = np.argmax(prediccion)

    # Retornar el resultado
    if etiqueta == 0:
        return "La imagen es una pera"
    else:
        return "La imagen es una manzana"
    
    ruta_imagen = "/content/Peras/Peras12.jpg"
resultado = probar_modelo(ruta_imagen)
print(resultado)