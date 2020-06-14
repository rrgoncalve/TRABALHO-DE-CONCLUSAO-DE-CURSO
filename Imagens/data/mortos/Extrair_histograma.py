import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

with open("inputs.csv",'w',newline='') as saida:
	escrever = csv.writer(saida)
	for i in range(1,13):
		print(str(i)+".jpg")
		img = cv2.imread(str(i)+".jpg")
		matriz = plt.hist(img.ravel(),256,[0,256])
		plt.show()
		escrever.writerow(matriz[0])
		print(matriz[0])
		
		


