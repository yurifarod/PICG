#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:28:27 2019
s
@author: yurifarod
"""

import numpy as np
import math
from matplotlib import pyplot as plt

def rgb2gray(img):
	return np.dot(img[:,:,:3],[.299,.587,.144]).astype('float')
   
def nativeDFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(121)
    plt.imshow(img, cmap = 'gray')
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def myDFT(img):
    shape = np.shape(img)
    saida = rgb2gray(img)
    entrada = rgb2gray(img)
    n = shape[1]
    m = shape[0]
    
    for u in range(0, m):
        for v in range(0, n):
            for x in range(0, m):
                for y in range(0, n):
                    saida[u][v] = entrada[x][y] * (math.cos( 2*math.pi*(u*x/m + v*y/n) ) - 1j * math.sin( 2*math.pi*(u*x/m + v*y/n) ) )
    return saida

def myInvertDFT(img):
    shape = np.shape(img)
    entrada = img
    saida = img
    n = shape[1]
    m = shape[0]
    
    for u in range(0, m):
        for v in range(0, n):
            for x in range(0, m):
                for y in range(0, n):
                    saida[u][v] = entrada[x][y] * (math.cos( 2*math.pi*(u*x/m + v*y/n) ) + 1j * math.sin( 2*math.pi*(u*x/m + v*y/n) ) )
    return saida
    

rgb_img = plt.imread('./Imagens/cancer.jpg')
gs_img = rgb2gray(rgb_img)
nativeDFT(gs_img)
    
#meuDFT = myDFT(rgb_img)
#plt.imshow(meuDFT)