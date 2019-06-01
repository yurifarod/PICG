# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def imsave(img):
	mpimg.imsave("tmp.png", img)

def imread(image_file):
	return mpimg.imread(image_file)


def abra(img2):
	img = imread(img2)
	plt.imshow(img, origin='image')
	plt.show()


def nchannels(img):
	dim = np.ndim(img)
	if dim == 3:
		return 3
	else:
		return 1


def size(img):
	shape = np.shape(img)
	width = shape[1]
	height = shape[0]
	return np.array([width,height])

def rgb2gray(img):
	return np.dot(img[:,:,:3],[.299,.587,.144]).astype('float')

def imshow(img):
	isRgb = nchannels(img) == 3
	minW = 50
	minH = 50

	if(isRgb):
		plt.imshow(img, interpolation='nearest')
	else:
		plt.imshow(img, cmap = plt.cm.gray, interpolation='nearest' )

	plt.show()
	return True


def thresh(img, l):
    return (img > l).astype(np.uint8) * 255

def neg(img):
    return 255-img

def contrast(img, r, m):
    return r*(img - m) + m

def histeq(img):
    nk = hist(img, 255).astype('float')
    n = sum(nk)
    pr = nk/n
    tr = np.array(range(0,255)).astype('float')
    tr[0] = pr[0]
    for k in range(1,255):
        tr[k] = pr[k]+tr[k-1]
    t255 = (tr * 255).astype(np.uint8)

    width,height = size(img)
    img2 = np.copy(img)
    img2 = t255[img]
    
    plt.clf()
    return img2

def convolve(img, mask):
    maskW, maskH = size(mask)
    mW = maskW % 2
    mH = maskH % 2
    width,height = size(img)
    img2 = np.copy(img)
    for x in range(0,height):
        for y in range(0,width):
            tot = 0
            for mx in range(-mH,mH):
                for my in range(-mW,mW):
                    if(x < mH):
                        val = img[x][y]
                    elif(y < mW):
                        val = img[x][y]
                    else:
                        val = img[x+mx][y+my]
                    tot = tot + (mask[mx+mH][my+mW]*val)

            img2[x][y] = tot
    return img2


#!!!!COMPUTAÇÃO GRÁFICA!!!!
def newImage(size, color):
    width, height = size
    img = np.array([[color for k in range(width)] for l in range(height)], dtype='uint8')
    return img

def drawDot(img, a, color):
    imW, imH = size(img)
    x = np.round_(a[0])
    y = np.round_(a[1])

    if (x >= 0 and x < imW) and (y >= 0 and y < imH):
        img[x][y] = color
    return img

def drawLine(img, a, b, color):
    x0, y0 = a
    x1, y1 = b
    dx = x1 - x0
    dy = y1 - y0
    x = x0
    y = y0

    if (abs(dx) > abs(dy)):
        step = abs(dx)
    else:
        step = abs(dy)

    if(step == 0):
        step = 1

    xInc = dx/step
    yInc = dy/step

    drawDot(img, [x,y], color)
    for i in range(np.round_(step).astype(int)):
        x += xInc
        y += yInc
        drawDot(img, [x,y], color)
    return img


def drawCircle(img, c, r, color):
    x0, y0 = c

    for x in range(r):
        y = np.sqrt((r*r) - (x*x))
        drawDot(img, [x0 -x,y0 -y], color)
        drawDot(img, [x0 -y,y0 -x], color)
        drawDot(img, [x0 +x,y0 -y], color)
        drawDot(img, [x0 -x,y0 +y], color)
        drawDot(img, [x0 +y,y0 -x], color)
        drawDot(img, [x0 -y,y0 +x], color)
        drawDot(img, [x0 +x,y0 +y], color)
        drawDot(img, [x0 +y,y0 +x], color)
        if(x > y): break

    return img


def drawPoly(img, dots, color):
    lastDot = dots[len(dots)-1]

    for i, dot in enumerate(dots):
        drawLine(img, lastDot, dot, color)
        lastDot = dot
    return img


#Execução
img = imread('./Imagens/cancer.jpg')
imgBlack = newImage([32,32],255)
imgWhite = newImage([32,32],0)
imgColored = newImage([500,500],[255,0,255])
imgCinza = rgb2gray(img)
imgNeg = neg(img)

dots = [[250, 250], [350, 250], [350, 350], [300, 350], [325, 300]]

#imshow(img)
#imshow(imgBlack)
#imshow(imgWhite)
imshow(imgColored)
imshow(imgCinza)
imshow(imgNeg)