import cv2
import os
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create()

def getImageWithId():
    '''
        Percorrer diretorio fotos, ler todas imagens com CV2 e organizar
        conjunto de faces com seus respectivos ids
    '''
    pathsImages = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []

    for pathImage in pathsImages:
        imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(pathImage)[-1].split('.')[1])

        ids.append(id)
        faces.append(imageFace)

        cv2.imshow("Face", imageFace)
        cv2.waitKey(10)
    return np.array(ids), faces


ids, faces = getImageWithId()

# Gerando classifier do treinamento
print("Treinando....")
lbph.train(faces, ids)

# Obter as previsões e as classes reais
predicted = []
actual = []
for i in range(len(ids)):
    label, confidence = lbph.predict(faces[i])
    predicted.append(label)
    actual.append(ids[i])

# Calcular a acurácia do modelo
matches = np.equal(predicted, actual)
num_matches = np.sum(matches)
accuracy = 100.0 * num_matches / len(ids)

# Imprimir a acurácia
print('Acurácia: %.2f%%' % accuracy)

lbph.write('classifier/classificadorLBPH.yml')
print('Treinamento concluído com sucesso!')