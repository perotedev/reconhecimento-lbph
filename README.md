# Reconhecimento Facial com OpenCV e LBPH

### Instalação

Crie um virtual env para empacotar suas libs Python

```
python -m venv venv
```

Ative sua virtual env

```
# para linux
source venv/bin/active

# para windows
.\venv\Scripts\activate
```

Atualize o PIP, gerenciador de pacotes do Python

```
python -m pip install --upgrade pip
```

Use o PIP, para instalar todos os requisitos

```
pip install -r requirements.txt
```

## Uso do programa

1 - Primeiro faça as capturas salvando as imagens das faces detectadas:

```
python capturar_face.py
```
digite um numero para ser o identificador da face e clique na tecla "q" para salvar a imagem da face detectada.


2 - Faça o apredizado das faces detectadas:

```
python treinamento.py
```
3 - Execute o teste para obter acurácia e matriz de confusão

```
python teste.py
```
4 - Execute o reconhecedor facil

```
python reconhecedor_lbph.py
```
<br>
Caso queira adicionar mais fotos ao modelo sem ter que treiná-lo por completo novamente

```
python atualizar.py
```


