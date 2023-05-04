import cv2
import os
import numpy as np

# Caminho Haarcascade
cascPath = 'cascade/haarcascade_frontalface_default.xml'
cascPathOlho = 'cascade/haarcascade-eye.xml'

# Classifier baseado nos haarcascade
facePath = cv2.CascadeClassifier(cascPath)
facePathOlho = cv2.CascadeClassifier(cascPathOlho)
video_capture = cv2.VideoCapture(0)

increment = 1
numMostras = 100
id = input('Digite seu identificador: ')
width, height = 220, 220
print('Capturando as faces...')

# Create directory para salvar on images
if not os.path.exists('fotos'):
    os.makedirs('fotos')

# mudar o valor do diretório conforme a necessidade
pathsImages = [os.path.join('fotos_thales', f) for f in os.listdir('fotos_thales')]

for pathImage in pathsImages:
    imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)

    # Realizando face detect
    face_detect = facePath.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minSize=(35, 35),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in face_detect:
        # Desenhando retangulo na face detectada
        cv2.rectangle(imageFace, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Realizando deteccao do olho da face
        region = imageFace[y:y + h, x:x + w]
        imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        face_detect_olho = facePathOlho.detectMultiScale(imageOlhoGray)

        face_off = cv2.resize(gray[y:y + h, x:x + w], (width, height))
        cv2.imwrite('fotos/pessoa.' + str(id) + '.' + str(increment) + '.jpg', face_off)

        print('[Foto ' + str(increment) + ' capturada com sucesso] - ', np.average(gray))
        increment += 1

    if increment > len(pathsImages): break

print('Fotos capturadas com sucesso :)')