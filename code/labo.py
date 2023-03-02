import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from zplane import zplane


def prob1():
    K = 1
    z1 = 0.8j
    z2 = -0.8j
    p1 = 0.95*np.exp((np.pi*1j)/8)
    p2 = 0.95*np.exp((-np.pi*1j)/8)

    num = np.poly([z1, z2])
    denum = np.poly([p1, p2])

    plot_filter(num, denum)

    impulsion = np.zeros(500)
    impulsion[int(len(impulsion)/2)] = 1
    plt.stem(impulsion)
    plt.show()

    h = filter_h(num, denum, impulsion, True)
    h_inv = filter_h(denum, num, h)
    plt.stem(h_inv)
    plt.show()


def filter_h(num, denum, signal, plot=False):
    h = signal.lfilter(num, denum, signal)
    
    if plot:
        plt.plot(h)
        plt.show()

    return h


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


def prob2():
    pi_16 = np.pi/16
    z = [ np.exp(1j*pi_16), np.exp(1j*(-pi_16)) ] 
    p = [ 0.95*np.exp(1j*pi_16), 0.95*np.exp(1j*(-pi_16)) ] 

    num = np.poly(z)
    denum = np.poly(p)

    plot_filter(num, denum)

    N = 500
    n = np.arange(0, N)
    x = [(np.sin(np.pi * n[i] / 16) + np.sin(np.pi * n[i] / 32)) for i in range(N)]

    plt.stem(x)
    plt.show()

    filtered = signal.lfilter(num, denum, x)

    plt.stem(filtered)
    plt.show()


def prob3():
    fe = 48000
    wp = 2500 / (fe/2)
    ws = 3500 / (fe/2)
    gstop = 40
    gpass = 0.2  

    butter_filter_conception(wp, ws, gstop, gpass)
    cheb1_filter_conception(wp, ws, gstop, gpass)
    cheb2_filter_conception(wp, ws, gstop, gpass)
    ellip_filter_conception(wp, ws, gstop, gpass)
    

def butter_filter_conception(wp, ws, gstop, gpass):
    N, wn = signal.buttord(wp, ws, gpass, gstop)    
    num, denum = signal.butter(N, wn)

    print_filter_order(N, wn, 'Butterworth')
    plot_filter(num, denum)


def cheb1_filter_conception(wp, ws, gstop, gpass):
    N, wn = signal.cheb1ord(wp, ws, gpass, gstop)    
    num, denum = signal.cheby1(N, gpass, wn)

    print_filter_order(N, wn, 'Chebyshev type I')
    plot_filter(num, denum)


def cheb2_filter_conception(wp, ws, gstop, gpass):
    N, wn = signal.cheb2ord(wp, ws, gpass, gstop)    
    num, denum = signal.cheby2(N, gstop, wn)

    print_filter_order(N, wn, 'Chebyshev type II')
    plot_filter(num, denum)


def ellip_filter_conception(wp, ws, gstop, gpass):
    N, wn = signal.ellipord(wp, ws, gpass, gstop)    
    num, denum = signal.ellip(N, gpass, gstop, wn)

    print_filter_order(N, wn, 'Elliptique')
    plot_filter(num, denum)


def print_filter_order(N, wn, name):
    print('{:20s} Ordre :  {:2d}     Wn : {:7.7f}'.format(name, N, wn))


def plot_filter(num, denum):
    rep_freq(num, denum)
    zplane(num, denum)


def prob4():
    plt.gray()
    img = plt.imread("images\goldhill.png")
    plt.imshow(img)
    plt.show()

    l, c = img.shape
    fact1 = 2
    fact2 = 1/2
    m = np.zeros([int(l*fact2), int(c*fact1)])
    T = [[fact1, 0], [0, fact2]]

    for x in range(c):
        for y in range(l):
            mat_res = np.matmul(T, [x, y])
            c_t = int(mat_res[0])
            l_t = int(l*fact2) - 1 - int(mat_res[1])
            m[l_t][c_t] = img[l - 1 - y, x] 

    # for x in range(c):
    #     for y in range(l):
    #         y_img = l- 1 - y
    #         res = np.dot(T, [x, y_img])
    #         c_t = int(res[0])
    #         l_t = int(res[1])
    #         m[l_t][c_t] = img[y_img][x]
    
    plt.imshow(m)
    plt.show()       


if __name__ == '__main__':
    # prob1()    
    # prob2()
    # prob3()
    prob4()