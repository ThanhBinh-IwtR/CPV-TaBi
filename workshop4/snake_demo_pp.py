import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from pylab import*
from typing import Any

class Snack:
    
    def __init__(self, raw_image, threshold=5):
        self.raw_image = raw_image
        self.threshold = threshold
        image_changed = self.change_type_image()
        self.snake(image_changed)

    def change_type_image(self):
        image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        image = np.array(image, dtype=np.float64)
        return image
    
    def mat_math(self, image_changed, intput, str):
        output=intput 
        for i in range(image_changed.shape[0]):
            for j in range(image_changed.shape[1]):
                if str=="atan":
                    output[i,j] = math.atan(intput[i,j]) 
                if str=="sqrt":
                    output[i,j] = math.sqrt(intput[i,j]) 
        return output 

    def chan_verse(self, LSF, image_changed, mean, nu, epsilon, step):
        Drc = (epsilon / math.pi) / (epsilon*epsilon+ LSF*LSF)
        Hea = 0.5*(1 + (2 / math.pi)*self.mat_math(image_changed, LSF/epsilon,"atan")) 
        Iy, Ix = np.gradient(LSF) 
        s = self.mat_math(image_changed, Ix*Ix+Iy*Iy,"sqrt") 
        Nx = Ix / (s+0.000001) 
        Ny = Iy / (s+0.000001) 
        Mxx, Nxx =np.gradient(Nx) 
        Nyy, Myy =np.gradient(Ny) 
        cur = Nxx + Nyy 
        Length = nu*Drc*cur 

        Lap = cv2.Laplacian(LSF,-1) 
        Penalty = mean*(Lap - cur) 

        s1 = Hea*image_changed 
        s2 = (1-Hea)*image_changed 
        s3 = 1-Hea 
        C1 = s1.sum()/ Hea.sum() 
        C2 = s2.sum()/ s3.sum() 
        CVterm = Drc*(-1 * (image_changed - C1)*(image_changed - C1) + 1 * (image_changed - C2)*(image_changed - C2)) 

        LSF = LSF + step*(Length + Penalty + CVterm) 
        return LSF

    def snake(self, image_changed):
        mu = 1
        nu = 0.003 * 255 * 255
        epison = 1
        step = 0.1
        weight, height = image_changed.shape
        IniLSF = np.ones((weight, height), image_changed.dtype)
        IniLSF[30:80,30:80] = -1
        IniLSF = -IniLSF 
        LSF=IniLSF 

        image_with_contour = np.copy(self.raw_image)
        for _ in range(1, self.threshold):
            LSF = self.chan_verse(LSF, image_changed, mu, nu, epison, step) 
            contours, hierarchy = cv2.findContours(LSF.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_with_contour, contours, -1, (0, 0, 255), 2)
        cv2.imshow("Image with contour", image_with_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()