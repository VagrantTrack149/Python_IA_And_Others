import numpy as np
from PIL import Image

# Cargar imagen y convertirla a escala de grises
img = Image.open("D:/Descargas/2_.jpg").convert("L")
img = np.array(img)

# Preprocesar la imagen
img = img.reshape(1, 28, 28) / 255.0

# Hacer predicción
pred = model.predict(img)

# Obtener número predicho
numero_predicho = np.argmax(pred)

print("El número predicho es:", numero_predicho)