#Clasificador de Imágenes de Pera o Manzana

#Primero, debemos importar las librerías necesarias:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Luego, debemos cargar las imágenes de pera y manzana. Supongamos que las imágenes están en dos carpetas diferentes, una para peras y otra para manzanas.

peras = glob.glob("peras/*.jpg")
manzanas = glob.glob("manzanas/*.jpg")

imagenes = peras + manzanas
etiquetas = [0] * len(peras) + [1] * len(manzanas)

# Ahora, debemos procesar las imágenes y convertirlas en un formato que pueda ser utilizado por la red neuronal. En este caso, vamos a resizear las imágenes a 32x32 pixels y convertirlas a escala de grises.

def procesar_imagen(address):
    img_width = 32
    img_height = 32
    rgb = io.imread(address)
    rgb_resized = resize(rgb, (img_height, img_width), anti_aliasing=True)
    gray_resized = img_as_ubyte(rgb2gray(rgb_resized))
    return gray_resized

imagenes_procesadas = [procesar_imagen(img) for img in imagenes]

# Luego, debemos dividir las imágenes en conjuntos de entrenamiento y prueba.

X = np.array(imagenes_procesadas)
y = np.array(etiquetas)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ahora, podemos crear el modelo de red neuronal convolucional.

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding="same"))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Conv2D(256, kernel_size=3, activation='relu', padding="same"))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Finalmente, podemos entrenar el modelo con los conjuntos de entrenamiento y prueba.

hist = model.fit(X_train, to_categorical(y_train), validation_data=(X_test, to_categorical(y_test)), epochs=300)

# Una vez entrenado el modelo, podemos utilizarlo para clasificar nuevas imágenes de pera o manzana.


# Crea un clasificador de imágenes que clasifique si una imagen es una pera o una manzana, para esto necesito que utilices redes neuronales convoluciones y que el codigo sea desarrollado en Python. Adicionalmente también necesito que crees una funcion para probar el modelo ya entrenado