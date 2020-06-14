#Importando bibliotecas
from matplotlib import pyplot as plt
from Networkstatus import Network as netStatuas
import numpy as np
import string as s
import cv2
import csv

cont=np.zeros((1,4), dtype=np.float64)
entrada=np.zeros((256,1), dtype=np.float64)
cons=None
# Carregando YOLO pela classe dnn.readNet que permite criar e manipular redes neurais artificiais abrangentes.
net = cv2.dnn.readNet("classifier/yolov3.weights", "cfg/yolov3.cfg")
#Carregando as classes do classificador
classes = []
with open("data/class.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#Realiza a captura dos nomes das camadas da rede 
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#define uma cor para cada classe de objetos
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Carregando imagens para teste 
img = cv2.imread("imgTest/m1.jpg")#imagem cam 1 Digital
imgTermica= cv2.imread("imgTest/m1_.jpg")#imagem cam 2 Termograma

#img = cv2.imread("imgTest/m2.jpg")#imagem cam 1 Digital
#imgTermica= cv2.imread("imgTest/m2_.jpg")#imagem cam 2 Termograma

#img = cv2.imread("imgTest/m3.jpg")#imagem cam 1 Digital
#imgTermica= cv2.imread("imgTest/m3_.jpg")#imagem cam 2 Termograma

#img = cv2.imread("imgTest/m4.jpg")#imagem cam 1 Digital
#imgTermica= cv2.imread("imgTest/m_.jpg")#imagem cam 2 Termograma

##Garantindo que as duas imagens tenham o mesmo tamanho para demosntração da imagem sobreposta.
height, width, channels = img.shape#Pega os valores do tamanho da imagem
imgTermica=cv2.resize(imgTermica,(width,height))

imgDupla = cv2.addWeighted(imgTermica, 0.5, img, 0.5, 0)#cria imagem sobreposta
# estrutura dos objetos encontrados
class_ids = []#classes
confidences = []#acuracidade
boxes = []#localização

def classifierObject(img,imgTermica,height, width, channels,class_ids,confidences,boxes):

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)#passando imagem para detectação na rede do YOLO
    outs = net.forward(output_layers)#Saida da rede 

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 :
                     # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    classifierObjectRectangle(indexes,img,imgTermica,height, width, channels,class_ids,confidences)
    
def classifierObjectRectangle(indexes,img,imgTermica,height, width, channels,class_ids,confidences):
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):  
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            acuracia= str(round((confidences[i]),2)*100)+"% " 
            if label.strip() == 'person':
                label=acuracia+label
                color= colors[i]
                


                #Crop imagem infra
                imgNova= imgTermica[y:y+h, x:x+w]# variavel a ser utilizada com a nova imagem
                histo_Termico = plt.hist(imgNova.ravel(),256,[0,256])
                labelStatus= classifierObjectStatus(histo_Termico)#pasar o histograma da imagem e obtem o retorno

				#cv2.imwrite("person.jpg", imgNova)
                #nesse trecho de  codigo e contornado o objeto emcontrado.    
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x+5, y + 20), font, 0.5, color, 1)
                #imagem infra
                cv2.rectangle(imgTermica, (x, y), (x + w, y + h), color, 2)
                cv2.putText(imgTermica, labelStatus, (x+5, y + 35), font, 0.4, color, 1)

                cv2.rectangle(imgDupla, (x, y), (x + w, y + h), color, 2)
                cv2.putText(imgDupla, label, (x+5, y + 20), font, 0.5, color, 1)
                cv2.putText(imgDupla, labelStatus, (x+5, y + 35), font, 0.4, color, 1)
    


    resultado = np.zeros((100, 500, 3), dtype=np.uint8)
    cv2.rectangle(resultado, (0, 0), (500, 100), (255, 255, 255), -1)

    cv2.putText(resultado, "Vitimas nao classificadas: " +str(cont[0][0]),(10, 10*2+10), font,0.8, (0,0,0), 1)
    cv2.putText(resultado, "Vitimas vivas: " +str(cont[0][1]), (10, 25*2+10), font, 0.8, (0,0,0), 1)
    cv2.putText(resultado, "Vitimas mortas:" +str(cont[0][2]), (10, 40*2+10), font, 0.8, (0,0,0), 1)
    cv2.imshow("Resultado", resultado)

    cv2.imshow("Cam 1", img)
    cv2.imshow("Visao dupla", imgDupla)
    cv2.imshow("Cam 2", imgTermica)

def classifierObjectStatus(histo_Termico):

	for i in range(0,256):
		entrada[i][0]=histo_Termico[0][i]

	res=netStatuas.myNeuralNetworkFunction(entrada)

	if res == -1:
		cont[0][0]+=1
		return "Não Classificado"
	if res>0.5:
		cont[0][1]+=1
		return str(round(res*100,2)) +"% Vitima Viva"
	else:
		cont[0][2]+=1
		return str(round(res*100,2)) +"% Vitima Morta"
		
		    
def main():
    
    classifierObject(img,imgTermica,height, width, channels,class_ids,confidences,boxes)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
        main()