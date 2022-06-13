import cv2
import numpy as np
import graphicFunctions as gf
import functions as f
'''
Procesamiento en general:
    
1. Filtro de mediana (matlab: medfilt2 opencv: medianBlur). Otros filtros
    - blur()
    - GaussianBlur()
    - MedianBlur()
    - BilateralBlur()
    
    Median Blur es un filtro muy efectivo para la eliminacion de ruido "sal y pimienta", 
    este filtro permite conservar los bordes de una imágen después de filtrar
    el ruido

2. Cambio de espacio de color
    - Se lleva los espacios de color CMYK y HSV
    - Se muestra Y y S para identificar con cuál es má fácil extraer la región

3. Umbralización
    - Umbralización de la imágen en escala de grises utilizando el método OTSU
    - OTSU se encarga de minimizar la varianza de los píxeles blancos y negros

4. Transformaciones morfológicas
    - Close: dilatación + erosión, sirve para cerrar o rellenear huecos
    - Open: erosión + dilatación, sirve para eliminar basura o ruido
'''

def bwareaopen(img, min_size, connectivity=8):
    """Remove small objects from binary image (approximation of 
    bwareaopen in Matlab for 2D images).

    Args:
        img: a binary image (dtype=uint8) to remove small objects from
        min_size: minimum size (in pixels) for an object to remain in the image
        connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

    Returns:
        the binary image with small objects removed
    """

    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    
    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]
        
        # remove connected components smaller than min_size
        if label_size < min_size:
            img[labels == i] = 0
            
    return img

archivo = open('../data/data.csv', 'a+')
archivo.write('imgName,area,convexArea,perimeter,maEllipse,MaEllipse,diameter,eccentricity\n')

for i in range(1,9):
    
    file = '../Images/Lunares/'+str(i)+'.jpg'
    img = cv2.imread(file)
    img = cv2.resize(img, (420,220)) # Redimensionado del frame
    
    filteredImg = cv2.medianBlur(cv2.medianBlur(cv2.medianBlur(img,3),3),3)
    
    cv2.imshow("Original vs Filtered Image", np.concatenate((img,filteredImg), axis=1))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # -------------------------------------------------------------------------------------------------- REPRESENTACION HSV
    
    h,s,v = cv2.split(cv2.cvtColor(filteredImg,cv2.COLOR_BGR2HSV))
    
    #hsv_img = cv2.cvtColor(filteredImg, cv2.COLOR_BGR2HSV)
    
    
    bgrdash = filteredImg.astype(np.float64)/255.
    # Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
    K = 1 - np.max(bgrdash, axis=2)
    # Calculate C
    C = (1-bgrdash[...,2] - K)/(1-K)
    # Calculate M
    M = (1-bgrdash[...,1] - K)/(1-K)
    # Calculate Y
    Y = (1-bgrdash[...,0] - K)/(1-K)
    # Combine 4 channels into single image and re-scale back up to uint8
    # CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)
    
    
    Ybw = (Y*255).astype(np.uint8)
    
    #cv2.imshow("S (hsv) and Y(cmyk)",  np.concatenate((s,Ybw), axis=1))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # --------------------------------------------------------------------------------------------------- BINARIZACION OTSU
    
    # -------- Binarizacion por medio del metodo Otsu, el cual escoge un valor medio que reduzca la varianza
    # -------- es el método que se implementa la funcion threshold de matlab
    
    _, yThresholded = cv2.threshold(Ybw,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    _, sThresholded = cv2.threshold(s,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    cv2.imshow('Thresholded Images', np.concatenate((yThresholded, sThresholded), axis=1))
    
    ## https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5,5),np.uint8)
    
    # Erosion (quitar basura externa)
    #erosion = cv2.erode(yThresholded,kernel,iterations = 2)
    
    # Dilatacion (Rellenar huecos y recuperar tamanio)
    #dilation = cv2.dilate(erosion,kernel,iterations = 2)

    morphology = cv2.morphologyEx(yThresholded, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_OPEN, kernel, iterations=3)
    
    
    #cv2.imshow('Close and Opening', dilation)
    
    # ----------------------------------------------------------------------------- RELLENADO DE HUECOS, TENCNICA FLOODFILL
    
    # Copy the thresholded image.
    # im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    #h, w = closedOpened.shape[:2]
    #mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Floodfill from point (0, 0) - Rellena los puntos conectados con blanco, con flag por defecto que es de 4
    #cv2.floodFill(im_floodfill, mask, (0, 0), 255, flags=8);
    
    # Invert floodfilled image - Imagen invertida
    #im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    #im_out = im_th | im_floodfill_inv
    
    # Display images.
    #cv2.imshow("Relleno de huecos", im_out)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # --------------------------------------------------------------------------------- ELIMINACION DE ELEMENTOS EXTERIORES
    
    # add 1 pixel white border all around (se busca eliminar todo lo conectado con el borde creado)
    
    pad = cv2.copyMakeBorder(morphology, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)
    #cv2.imshow("Image With Border", pad)
    
    h, w = pad.shape
    
    # create zeros mask 2 pixels larger in each dimension, la suma de 2 a cada dimension, es por formula practicamente
    mask = np.zeros([h + 2, w + 2], np.uint8)
    
    # floodfill pone en negro todo lo que se conecto con el borde creado
    # floodfill params:
    #   imagen a procesar
    #   mascara para rellenado
    #   punto de partida o de origen de aplicacion del algoritmo (0,0)
    #   nuevo valor a aplicar a los pixeles conectado al punto de partida u origen
    #   Tolerancia minima en el brillo o color para el rellenado
    #   Tolerancia maxima en el brillo o color para el rellenado
    #   Flags, son la cantidad de pixeles alrededor para aplicar el algoritmo que rellena de cierto color
    
    img_floodfill = cv2.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1]
    
    
    # remove border and closing
    img_floodfill = cv2.morphologyEx(img_floodfill[1:h-1, 1:w-1], cv2.MORPH_CLOSE, kernel, iterations=2)
    #cv2.imshow("Floodfill image REMOVE BORDER", img_floodfill)
    
    #cv2.imshow("Floodfill image", img_floodfill)
    
    #Area open to delete small objects 
    bwareaopen(img_floodfill, 1500)
    
    extracted = cv2.bitwise_and(filteredImg, filteredImg, mask=img_floodfill)
        
    # show the images
    cv2.imshow("Extracted", extracted)
    
    key = cv2.waitKey(0)
    
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    elif  key == ord('c'):
        cv2.destroyAllWindows()
    else:
        try:
            image = img_floodfill
            _, _, stats, centroids = cv2.connectedComponentsWithStats(image) #Se hace una segmentación para la identificación de los objetos
        
            #Stats contiene: Leftmost (x), Topmost(y), Width, Height y área(pixeles) de cada objeto
            #centroids contiene los centroides de cada uno de los objetos
        
            stats = np.delete(stats, 0, 0)              #elimina las estadisticas del primer label que es el fondo
            centroids = np.delete(centroids, 0, 0)      #elimina el centroide del primer label que es el fondo
            
            if len(centroids) != 1:
                print('None or more than 1 object are detected')
            
            img2show = np.concatenate((gf.brect(image, stats), gf.area(image, stats), gf.convex_hull(image)), axis=1)
            img2show = np.concatenate((img2show, np.concatenate((gf.convex_area(image), gf.convex_image(image), gf.perim(image, centroids)), axis=1)), axis=0)
            img2show = np.concatenate((img2show, np.concatenate((gf.ellips(image), gf.eccent(image), gf.equivD(image, stats)), axis=1)), axis=0)
            
            cv2.imshow('Characteristics', img2show)
            cv2.waitKey(0)    
            cv2.destroyAllWindows()
            
            archivo.write(file.split('/')[-1]+","+f.getAllFeatures(image)+'\n')
            
            
        except:
            print('Error en la extracción de las características')
            
    #print("Getting all features: (area + convexArea + perimeter + ellipse + diameter + eccentricity) \n\n", f.getAllFeatures(image),"\n\n", 45*"-")

archivo.close()
    
    
