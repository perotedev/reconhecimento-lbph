import cv2
import os
import numpy as np

def getImageWithId():
    '''
        Percorrer diretorio fotos, ler todas imagens com CV2 e organizar
        conjunto de faces com seus respectivos ids
    '''
    pathsImages = [os.path.join('fotos1', f) for f in os.listdir('fotos1')]
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

# Usando algoritmos de face detect
lbph = cv2.face.LBPHFaceRecognizer_create()

# Instanciado Faces Recognizer e atualizando
lbph.read("classifier/classificadorLBPH.yml")

# Gerando classifier do treinamento
print("Treinando....")
lbph.update(faces, ids)
lbph.write('classifier/classificadorLBPH.xml')
print('Treinamento concluído com sucesso!')