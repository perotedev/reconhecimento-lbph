import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.read("classifier/classificadorLBPH.yml")

arrayFaces = [
    "01 - Rodrigo",
    "02 - Airton",
    "03 - Angel",
    "04 - Aurelio",
    "05 - Yan",
    "06 - Pessoa_t1",
    "07 - Pessoa_t2",
    "08 - Thalles",
    "09 - Pessoa_t3",
    "10 - Kail",
    "11- Yasmin"
]

def getImageWithId():
    '''
        Percorrer diretorio fotos, ler todas imagens com CV2 e organizar
        conjunto de faces com seus respectivos ids
    '''
    pathsImages = [os.path.join('fotos_teste', f) for f in os.listdir('fotos_teste')]
    faces = []
    ids = []
    labels= []

    for pathImage in pathsImages:
        imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(pathImage)[-1].split('.')[1])
        label = arrayFaces[id-1]

        ids.append(id)
        faces.append(imageFace)
        labels.append(label)

        cv2.imshow("Face", imageFace)
        cv2.waitKey(10)
    return labels, np.array(ids), faces


labels, ids, faces = getImageWithId()

# Obter as previsões e as classes reais
predicted = []
actual = []
labels_actual = []
for i in range(len(ids)):
    label, confidence = lbph.predict(faces[i])
    predicted.append(label)
    actual.append(ids[i])
    labels_actual.append(labels[i])

# Calcular a acurácia do modelo
matches = np.equal(predicted, actual)
num_matches = np.sum(matches)
accuracy = 100.0 * num_matches / len(ids)

# Imprimir a acurácia
print('Acurácia: %.2f%%' % accuracy)

# Criar a matriz de confusão
cm = confusion_matrix(actual, predicted)

# Imprimir a matriz de confusão
print("Matriz de confusão:")
print(cm)

# Configurar o plot da matriz de confusão
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=np.unique(labels_actual),
       yticklabels=np.unique(labels_actual),
       xlabel='Predição',
       ylabel='Real')

# Rotacionar os rótulos dos eixos x e ajustar a posição do texto
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2. else "black")

# Configurar o título e exibir o plot
ax.set_title("Matriz de Confusão")
fig.tight_layout()
plt.show()
