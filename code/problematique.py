import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from zplane import zplane


def read_img(filename, png=False):
    filename = f"images\{filename}"
    return np.load(filename) if not png else plt.imread(filename)


def show_img(img, name):
    plt.imshow(img)
    plt.title(name)
    plt.show()


def rep_freq(num, denum):
    w, amp = signal.freqz(num, denum)

    fig, ax1 = plt.subplots()
    ax1.set_title('Amplitude et phase de la réponse en fréquence')
    ax1.plot(w, 20 * np.log10(abs(amp)), 'r')
    ax1.set_xlabel('Fréquence (rad/éch.)')
    ax1.set_ylabel('Amplitude (dB)', color='r')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(amp))
    ax2.plot(w, angles, 'b')
    ax2.set_ylabel('Angle (rad)', color='b')
    plt.show()


def get_poles_zeros_frac(z, p, plot=False):
    num = np.poly(z)
    denum = np.poly(p)

    if plot:
        zplane(num, denum)

    return num, denum


def inv_filter(plot=False):
    z = [ 0.9*np.exp(1j*np.pi/2), 0.9*np.exp(-1j*np.pi/2) , 0.95*np.exp(1j*np.pi/8), 0.95*np.exp(-1j*np.pi/8) ] 
    p = [ 0, -0.99, -0.99, 0.8 ] 

    return get_poles_zeros_frac(p, z, plot)


def filter_img(img, num, denum, name, plot=False):

    img_filtered = signal.lfilter(num, denum, img)
    
    if plot:
        show_img(img_filtered, name)

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
    l = img.shape[0]
    c = img.shape[1]
    # T = rotation_matrix(l, c)
    T = [[0, 1], [-1, 0]]
    mat = np.zeros((l, c))

    for x in range(c):
        for y in range(l):
            y_img = l - 1 - y
            res = np.dot(T, [x, y_img])
            c_t = int(res[0])
            l_t = int(res[1])
            mat[l_t][c_t] = img[y_img][x]
    # for x in range(c):
    #     for y in range(l):
    #         # y_img = l - 1 - y
    #         res = np.dot(T, [x, y])
    #         c_t = int(res[0])
    #         l_t = l - 1 - int(res[1])
    #         # print(f"l - 1 - y : {l - 1 - y}")
    #         # print(f"l_t : {l_t}")
    #         # print(f"c_t : {c_t} int(res[1])  {int(res[1]) }")
    #         # print(f"x : {x}")

    #         mat[l_t][c_t] = img[l - 1 - y][x]

    if plot:
        show_img(mat, "Image avec rotation")

    return mat


def filter_conception_valid(plot=False):
    z = [ -1, -1 ]
    p = [ -0.2317 + 0.3948j, -0.2317 - 0.3948j ]

    return get_poles_zeros_frac(z, p, plot)


def low_pass_filter_conception(plot=False):
    fe = 1600
    wp = 500 / (fe/2)
    ws = 750 / (fe/2)
    gstop = 60
    gpass = 0.2  

    butter = butter_filter_conception(wp, ws, gstop, gpass, plot)
    cheb1 = cheb1_filter_conception(wp, ws, gstop, gpass, plot)
    cheb2 = cheb2_filter_conception(wp, ws, gstop, gpass, plot)
    ellip = ellip_filter_conception(wp, ws, gstop, gpass, plot)

    filters = sorted([ butter, cheb1, cheb2, ellip ], key=lambda x: x[0])

    return filters[0][1]


def butter_filter_conception(wp, ws, gstop, gpass, plot=False):
    N, wn = signal.buttord(wp, ws, gpass, gstop)    
    num, denum = signal.butter(N, wn)

    rep_filter(num, denum, N, wn, 'Butterworth', plot)
    return (N, (num, denum))


def cheb1_filter_conception(wp, ws, gstop, gpass, plot=False):
    N, wn = signal.cheb1ord(wp, ws, gpass, gstop)    
    num, denum = signal.cheby1(N, gpass, wn)

    rep_filter(num, denum, N, wn, 'Chebyshev type I', plot)
    return (N, (num, denum))


def cheb2_filter_conception(wp, ws, gstop, gpass, plot=False):
    N, wn = signal.cheb2ord(wp, ws, gpass, gstop)    
    num, denum = signal.cheby2(N, gstop, wn)

    rep_filter(num, denum, N, wn, 'Chebyshev type II', plot)
    return (N, (num, denum))


def ellip_filter_conception(wp, ws, gstop, gpass, plot=False):
    N, wn = signal.ellipord(wp, ws, gpass, gstop)    
    num, denum = signal.ellip(N, gpass, gstop, wn)

    rep_filter(num, denum, N, wn, 'Elliptique', plot)
    return (N, (num, denum))


def rep_filter(num, denum, N, wn, name, plot=False):
    if plot:
        rep_freq(num, denum)
        zplane(num, denum)
    print('{:20s} Ordre :  {:2d}     Wn : {:7.7f}'.format(name, N, wn))


def main():
    plt.gray()
    filename1 = 'image_complete.npy'
    filename2 = 'goldhill_aberrations.npy'
    filename3 = 'goldhill_rotate.png'
    
    img = read_img(filename1)
    # img = read_img(filename2)
    # img = read_img(filename3, True)
    show_img(img, "Image initiale")

    # Remove aberrations
    num_abr, denum_abr = inv_filter(True)
    img_no_abr = filter_img(img, num_abr, denum_abr,"Image sans aberrations", True)
    # Rotate image
    img_rotated = rotate_img(img_no_abr, True)

    # Remove noise (sans python)
    num, denum = filter_conception_valid(False)
    img_no_noise1 = filter_img(img_rotated, num, denum, "Image sans bruit (sans python)", True)

    # Remove noise (avec python)
    num, denum =  low_pass_filter_conception(False)
    img_no_noise2 = filter_img(img_rotated, num, denum, "Image sans bruit (avec python)",True)
    

if __name__ == '__main__':
    main()