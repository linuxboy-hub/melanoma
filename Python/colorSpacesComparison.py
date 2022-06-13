import cv2
import numpy as np

def BGR2CMYK(image, imageHsv):
    imageCopy = image.copy()
    h,s,v = cv2.split(imageHsv)
    
    bgrdash = imageCopy.astype(np.float)/255
    # Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
    K = 1 - np.max(bgrdash, axis=2)
    # Calculate C
    C = (1-bgrdash[...,2] - K)/(1-K)
    # Calculate M
    M = (1-bgrdash[...,1] - K)/(1-K)
    # Calculate Y
    Y = (1-bgrdash[...,0] - K)/(1-K)
    # Combine 4 channels into single image and re-scale back up to uint8
    CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)
    
    return CMYK
    

# Import image
for i in range(1,9):
    
    file = '../Images/Lunares/'+str(i)+'.jpg'
    
    img = cv2.imread(file)
    img = cv2.resize(img, (260,125)) # Redimensionado del frame
    
    b,g,r = cv2.split(img)
    imgBgr = np.concatenate((b,g,r), axis=1)
        
    
    # Extracting different color spaces of the image
    l,a,b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    imgLab = np.concatenate((l,a,b), axis=1)
    

    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    imgHsv = np.concatenate((h,s,v), axis=1)
    
    
    c,m,y,k = cv2.split(BGR2CMYK(img.astype(float)/255., cv2.cvtColor(img, cv2.COLOR_BGR2HSV)))
    imgCmyk = np.concatenate((c,m,y), axis=1)
    
    
    h,l,s = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    imgHls = np.concatenate((h,l,s), axis=1)
    
    
    l,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LUV))
    imgLuv = np.concatenate((l,u,v), axis=1)
    
    img2show = np.concatenate((imgLab,imgHsv), axis=1)
    img2show = np.concatenate((img2show,np.concatenate((imgCmyk,imgHls), axis=1)), axis=0)
    img2show = np.concatenate((img2show,np.concatenate((imgLuv,imgBgr), axis=1)), axis=0)
    
    img2show = cv2.cvtColor(img2show, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(img2show, 'Lab', (12,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 87, 255), 2, cv2.LINE_AA)
    cv2.putText(img2show, 'Hsv', (792,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 87, 255), 2, cv2.LINE_AA)
    cv2.putText(img2show, 'cmy', (12,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 87, 255), 2, cv2.LINE_AA)
    cv2.putText(img2show, 'hls', (792,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 87, 255), 2, cv2.LINE_AA)
    cv2.putText(img2show, 'luv', (12,275), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 87, 255), 2, cv2.LINE_AA)
    cv2.putText(img2show, 'Bgr', (792,275), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 87, 255), 2, cv2.LINE_AA)

    
    cv2.imshow('Image-view',img2show)
    cv2.imshow('Original Image-view',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()