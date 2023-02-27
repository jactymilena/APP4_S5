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


# def rotation_matrix(l, c, plot=False):
#     mat = np.zeros((l, c))

#     for i in range(c):
#         for j in range(l):
#             if i == l - j - 1:
#                 mat[i, j] = 1

#     if plot:
#         print(mat)

#     return mat


def rotate_img(img, plot=False):
    if plot:
        show_img(img)

    l = img.shape[0]
    c = img.shape[1]
    # T = rotation_matrix(l, c)
    T = [[0, 1], [1, 0]]
    mat = np.zeros((l, c))

    for x in range(c):
        for y in range(l):
            y_img = l - 1 - y
            res = np.dot(T, [x, y_img])
            c_t = int(res[0])
            l_t = int(res[1])
            mat[l_t][c_t] = img[y_img][x]

    show_img(mat)



def main():
    plt.gray()
    filename1 = 'image_complete.npy'
    filename2 = 'goldhill_aberrations.npy'
    filename3 = 'goldhill_rotate.png'

    img = read_img(filename1)
    # img = read_img(filename2)
    # img = read_img(filename3, True)

    num, denum = inv_filter()
    img_filtered = remove_aberrations(img, num, denum, True)
    rotate_img(img_filtered)

    


if __name__ == '__main__':
    main()