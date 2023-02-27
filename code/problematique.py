import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from zplane import zplane


def read_img(filename, png=False):
    filename = f"images\{filename}"
    return np.load(filename) if not png else plt.imread(filename)


def show_img(img):
    plt.imshow(img)
    plt.show()


def inv_filter(plot=False):
    z = [ 0, -0.99, -0.99, 0.8 ] 
    p = [ 0.9*np.exp(1j*np.pi/2), 0.9*np.exp(-1j*np.pi/2) , 0.95*np.exp(1j*np.pi/8), 0.95*np.exp(-1j*np.pi/8) ] 

    num = np.poly(z)
    denum = np.poly(p)
    if plot:
        zplane(num, denum)

    return num, denum

def remove_aberrations(img, num, denum, plot=False):

    img_filtered = signal.lfilter(num, denum, img)
    
    if plot:
        show_img(img)
        show_img(img_filtered)

    return img_filtered


def rotate_img(img):
    show_img(img)


def main():
    plt.gray()
    filename1 = 'image_complete.npy'
    filename2 = 'goldhill_aberrations.npy'
    filename3 = 'goldhill_rotate.png'

    # img = read_img(filename1)
    # img = read_img(filename2)
    img = read_img(filename3, True)

    num, denum = inv_filter()
    # img_filtered = remove_aberrations(img, num, denum, True)
    rotate_img(img)


if __name__ == '__main__':
    main()