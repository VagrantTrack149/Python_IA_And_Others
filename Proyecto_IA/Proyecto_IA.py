# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:59:20 2023

@author: Neil O
"""

# Importar librerias
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
# Función para cargar y etiquetar las imágenes de una carpeta
def load_images_from_folder(folder, label, target_size=(256, 256)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize the image to a common size
            images.append(img)
            labels.append(label)
    return images, labels
# Crear el analizador de argumentos
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-k', '--num-clusters', default=6, type=int, help='Número de clústeres = Valores entre 2 y 10')
arguments = argument_parser.parse_args()

# Cargar imágenes de las carpetas de entrenamiento
normal_images, normal_labels = load_images_from_folder('Training/NORMAL', label=0)
glaucoma_images, glaucoma_labels = load_images_from_folder('Training/Glaucoma', label=1)

# Concatenar las imágenes y etiquetas
training_images = normal_images + glaucoma_images
training_labels = normal_labels + glaucoma_labels

# Convertir a un formato adecuado para K-Means
pixel_values = np.vstack(training_images).reshape((-1, 3))
pixel_values = np.float32(pixel_values)
# Aplicar K-Means al conjunto de entrenamiento
kmeans = KMeans(n_clusters=arguments.num_clusters, random_state=0)
kmeans.fit(pixel_values)

# Ruta de la imagen de prueba
test_image_path = 'Test/normal/n_001.png'  # Reemplaza 'your_test_image.jpg' con el nombre de tu imagen

# Cargar la imagen de prueba
test_image = cv2.imread(test_image_path)
if test_image is not None:
    test_image = cv2.resize(test_image, (256, 256))  # Ajusta el tamaño si es necesario

    # Preparar la imagen para la segmentación
    test_pixel_values = test_image.reshape((-1, 3))
    test_pixel_values = np.float32(test_pixel_values)

    # Asignar etiquetas usando el modelo K-Means entrenado
    test_labels = kmeans.predict(test_pixel_values)

    # Contar las ocurrencias de cada etiqueta
    label_counts = np.bincount(test_labels)

    # Obtener la etiqueta más común
    most_common_label = np.argmax(label_counts)

    # Imprimir en la consola si es "Normal" o "Glaucoma"
    if most_common_label == 1:
        print("La imagen tiene Glaucoma.")
    else:
        print("La imagen es Normal.")
else:
    print("Error al cargar la imagen de prueba.")


# Capturar video desde la cámara
video_capture = cv2.VideoCapture(0)

# Lista para almacenar las imágenes capturadas
last_segmented_image = None

# Crear ciclo WHILE para ejecutar el programa hasta que el usuario desee.
while True:
    # Leer un frame del video
    ret, frame = video_capture.read()

    # Mostrar la imagen original
    cv2.imshow('Imagen original', frame)

    # Preparar la imagen para la segmentación
    pixel_values = frame.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Asignar etiquetas usando el modelo K-Means entrenado
    labels = kmeans.predict(pixel_values)

    # Contar las ocurrencias de cada etiqueta
    label_counts = np.bincount(labels)

    # Obtener la etiqueta más común
    most_common_label = np.argmax(label_counts)

    # Imprimir en la consola si es "Normal" o "Glaucoma"
    if most_common_label == 1:
        print("Glaucoma")
    else:
        print("Normal")
        
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(frame.shape)
    # Mostrar la imagen de prueba segmentada
    test_segmented_data = centers[test_labels.flatten()]
    test_segmented_image = test_segmented_data.reshape(test_image.shape)
    cv2.imshow('Imagen de prueba segmentada', test_segmented_image)

    # Mostrar la imagen segmentada
    cv2.imshow('Imagen segmentada', segmented_image)
    
    # Guardar la imagen capturada en la lista
    last_segmented_image = segmented_image

    # Salir del bucle si se presiona la tecla 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
    
    
# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()

# Guardar las imágenes capturadas en el disco
if last_segmented_image is not None:
    cv2.imwrite('captured_segmented_image.jpg', last_segmented_image)
