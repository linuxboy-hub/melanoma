"""
----------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------- Librerias necesarias ---------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------
"""
import cv2
import numpy as np
import random as rng
import math
import csv
import pandas as pd

"""
----------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------- Definición de funciones ------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------
"""

def getAllFeatures(imagen):
    
    _, _, stats, centroids = cv2.connectedComponentsWithStats(imagen) #Se hace una segmentación para la identificación de los objetos

    #Stats contiene: Leftmost (x), Topmost(y), Width, Height y área(pixeles) de cada objeto
    #centroids contiene los centroides de cada uno de los objetos

    stats = np.delete(stats, 0, 0)              #elimina las estadisticas del primer label que es el fondo
    centroids = np.delete(centroids, 0, 0)      #elimina el centroide del primer label que es el fondo
    
    area = getArea(imagen, stats)
    convexArea = getConvex_area(imagen)
    perimeter = getPerim(imagen, centroids)
    ellipse = getEllips(imagen)
    diameter = getEquivD(imagen, stats)
    eccentricity = getEccent(imagen)
    
    features = area+","+convexArea+","+perimeter+","+ellipse+","+diameter+","+eccentricity
    
    return features

    

"""---------- BoundigBox de los objetos -------------------"""
def getBrect(imagen, stats):
  img = imagen.copy()
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                 #Se pasa al espacio de color RGB para  dibujar el rectangulo con color diferenciador
  for i in stats:                                             #Stats contiene: Leftmost (x), Topmost(y), Width, Height y área(pixeles) de cada objeto
    x,y,w,h = i[0], i[1], i[2], i[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))   #cv2.rectangle(image, (top-left point), (bottom-right point), (color))
  return img


"""---------- Área de los objetos -------------------"""
def getArea(imagen, stats):
  img = imagen.copy()
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                 #Se pasa al espacio de color RGB para dibujar las estadisticas en color diferenciador
  for i in stats:
    objectArea = str(i[4])                                 #Stats contiene: Leftmost (x), Topmost(y), Width, Height y área(pixeles) de cada objeto
    cv2.rectangle(img,(int(i[0]), int(i[1])-15), (int(i[0]+len(str(i[4]))*9), int(i[1])+5),(178,114,45), cv2.FILLED)    #Rectangulo que funciona como fondo para el texto
    cv2.putText(img, str(i[4]), (int(i[0]), int(i[1])), 5, 0.7, (255, 255, 255))            #Se escribe el valor del area de cada objeto en la imagen
  return objectArea


"""---------- ConvexHull de los objetos -------------------"""
def getConvex_hull(imagen):
  img = imagen.copy()
  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   #Se buscan los contornos en la imagen binarizada
  hull_list = []                                              #Lista en la cual se guardan los convex hull
  for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])                        #Se calcula el contorno convexo en base al contorno del objeto
    hull_list.append(hull)                                    #Se anexa a la lista

  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                 #Se pasa al espacio de color RGB para dibujar el convex hull en color diferenciador
  for i in range(len(contours)):                              #Ciclo para recorrer la lista de convex hull y graficarlos
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))   #Se genera el color de manera aleatoria
    cv2.drawContours(img, hull_list, i, color,2)                              #funcion para graficar el contorno en especifico

  return img


"""---------- Área convexa de los objetos -------------------"""
def getConvex_area(imagen):
    img = getConvex_image(imagen.copy())                         #Se determina la imagen convexa de todos los objetos
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)               #Se pasa a escala de grises
    _, _, stats, _ = cv2.connectedComponentsWithStats(img)    #Se hallan las estadísticas de los objetos
    stats = np.delete(stats, 0, 0)                            #Se elimina las estadisticas del primer label que es el fondo
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)               #Se pasa a RGB para dibujar la estadistica en un color diferenciador
    for i in stats:                                           #Stats contiene: Leftmost (x), Topmost(y), Width, Height y área(pixeles) de cada objeto
      objectArea = str(i[4])
      cv2.rectangle(img, (int(i[0]), int(i[1]) - 15), (int(i[0] + len(str(i[4])) * 9), int(i[1]) + 5), (178, 114, 45),cv2.FILLED)
      cv2.putText(img, str(i[4]), (int(i[0]), int(i[1])), 5, 0.7, (255, 255, 255))

    return objectArea


"""---------- Imágen convexa de los objetos -------------------"""
def getConvex_image(imagen):
    img = imagen.copy()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #Se buscan los contornos externos en la imagen binarizada
    hull_list = []
    for i in range(len(contours)):
      hull = cv2.convexHull(contours[i])                            #Se calculan los contornos convexos de cada objeto
      hull_list.append(hull)                                        #Se anexa a la lista
    cv2.drawContours(img, hull_list,-1,(255,0), cv2.FILLED)         #Se grafican todos los contornos convexos rellenados por dentro (cv2.FILLED=-1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                     #Se pasa a RGB para poder ser mostrado en un QLabel

    return img



"""---------- Perímetro de los objetos -------------------"""
def getPerim(imagen,centroids):
  img = imagen.copy()
  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #Se buscan los contornos externos en la imagen binarizada
  perim_list = []
  for i in range(len(contours)):
    perim = cv2.arcLength(contours[i], True)                                    #Funcion para calcular la longitud del contorno, que es el perimetro del objeto
    perim_list.append(round(perim,2))                                           #Se anexa a una lista el valor redondeado en 2 cifras decimales del perimetro hallado
  j = 0                                                                         #Contador para recorrer la lista que contiene el valor de perimetros de los contornos
  color = (0, 255, 0)
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(img, contours,-1, color, 2)

  for i in centroids:
    cv2.rectangle(img, (int(i[0])-20, int(i[1]) - 15), (int(i[0]) + len(str(perim_list[j])) * 9 - 20, int(i[1]) + 5), (178, 114, 45), cv2.FILLED)
    cv2.putText(img, str(perim_list[j]), (int(i[0])-20, int(i[1])), 5, 0.7, (255, 255, 255))
    objectPerim = str(perim_list[j])
    j += 1
  return objectPerim


"""---------- Elipse aproximada para cada objeto -------------------"""
def getEllips(imagen):
  img = imagen.copy()
  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  for i in contours:
    ellip = cv2.fitEllipse(i)                                                       #Función que encuentra una elipse aproximada a un contorno específico
    cv2.ellipse(img, ellip, (0, 255, 0), 2)                                         #Se grafíca la elipse en la imágen
  #return "ma "+str(ellip[1][0])+", "+"MA "+str(ellip[1][1])
  return str(ellip[1][0])+","+str(ellip[1][1])


"""---------- Excentricidad de cada objeto -------------------"""
def getEccent(imagen):
  img = imagen.copy()
  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  for i in contours:
    ellip = cv2.fitEllipse(i)
    sma = ellip[1][0]/2                                   #Semi-eje menor
    SMA = ellip[1][1]/2                                   #Semi-eje mayor
    e = ((SMA**2-sma**2)**0.5)/SMA                        #Se calcula la excentricidad según la formula, en base a los semi-ejes de la elipse
    cv2.rectangle(img, (int(ellip[0][0] - 25), int(ellip[0][1]) - 15),(int(ellip[0][0] - 25) + len(str(round(e, 3))) * 9, int(ellip[0][1]) + 5), (178, 114, 45),cv2.FILLED)
    cv2.putText(img, str(round(e, 3)), (int(ellip[0][0] - 25), int(ellip[0][1])), 5, 0.7, (255, 255, 255))
    objectEccent = str(round(e, 3))
  return objectEccent


"""---------- Diámetro equivalente de cada objeto -------------------"""
def getEquivD(imagen,stats):
  img = imagen.copy()
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  for i in stats:
    radio = round(math.sqrt(4*i[4]/math.pi),2)     #Se calcula el radio en base al área del objeto, asumiendolo como una circunferencia, el resultado se redondea a dos decimales
    cv2.rectangle(img, (int(i[0]+i[2]/4), int(i[1]+3*i[3]/5) - 15),(int(i[0]+i[2]/4) + len(str(radio)) * 9, int(i[1]+3*i[3]/5) + 5), (178, 114, 45), cv2.FILLED)
    cv2.putText(img,str(radio),(int(i[0]+i[2]/4), int(i[1]+3*i[3]/5)), 5, 0.7, (255, 255, 255))
    objectDiam = str(radio)
  return objectDiam



"""---------- Lista de pixeles con valor mayor a cero en toda la imágen -------------------"""
def pixelList(imagen):
  pixels = cv2.findNonZero(imagen)              #Función que retorna una lista de los pixeles con valor mayor que 0
  #df = pd.DataFrame(list(zip(*pixels)))
  pixel_list = []
  for i in pixels:
    data = i[0][:]
    pixel_list.append(data)
  df = pd.DataFrame(pixel_list)
  df.to_csv('pixel_List.csv', index=False)



"""---------- Lista de pixeles de cada objeto -------------------"""
def pixelIdxList(imagen):
  objects_list = []
  img = imagen.copy()
  img2 = np.zeros(img.shape, dtype = "uint8")                       #Imágen negra de las mismas dimensiones de la imagen original
  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  for i in range(len(contours)):
      pixel_list = []
      cv2.drawContours(img2, contours, i, (255, 0), cv2.FILLED)
      img2 = cv2.bitwise_and(img, img2)                             #Se obtiene una imágen con solo un objeto graficado (img AND img2)
      pixels = cv2.findNonZero(img2)                                # Función que retorna una lista de los pixeles con valor mayor que 0
      for i in pixels:
        pixel_list.append(i[0][:])

      objects_list.append(pixel_list)  # Se anexa la lista de pixeles con valor mayor a cero del objeto a objects_list
      img2 = np.zeros(img.shape, dtype="uint8")

  with open('pixel_IdxList.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(objects_list)

