import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from PIL import Image


def get_fwhm(data, rfactor=1, plot=False):
    xdat = lineout_x(data)[::rfactor]
    ydat = lineout_y(data)[::rfactor]

    fwhm_x, xopt = fit_Gaussian(np.linspace(0, len(xdat), len(xdat)), xdat, plot=plot, interp=False)
    fwhm_y, yopt = fit_Gaussian(np.linspace(0, len(ydat), len(ydat)), ydat, plot=plot, interp=False)
    xopt = [xopt[0], xopt[1] * rfactor, xopt[2] * rfactor]
    yopt = [yopt[0], yopt[1] * rfactor, yopt[2] * rfactor]

    return [fwhm_x * rfactor, fwhm_y * rfactor], [xopt, yopt]


def lineout_y(data):
    avg_arr = []
    for i in range(len(data)):
        total = 0
        for j in range(len(data[i])):
            total += data[i][j]
        avg_arr.append(total / len(data[i]))
    #print('FWHM from un-normalized std: ', np.std(avg_arr))
    return normalize(avg_arr)

def lineout_x(data):
    avg_arr = []
    length = len(data)
    # assumes all sub arrays are the same shape - image is always 1280x1024 so this assumption is valid.
    for j in range(len(data[0])):
        total = 0
        for i in range(len(data)):
            total += data[i][j]
        avg_arr.append(total / length)
    #print('FWHM from un-normalized std: ', np.std(avg_arr))
    return normalize(avg_arr)


def normalize(iarr):
    ret = []
    mx = max(iarr)
    mn = min(iarr)
    for i in iarr:
        ret.append(np.interp(i, [mn, mx], [0, 1]))
    return ret


def fit_Gaussian(xi, data, plot=True, interp=True):
    """
    does... stuff.
    :param plot: Whether or not to plot data or just return numerical values
    :param xi: x coordinates in a numpy array
    :param data: y coordinates in a numpy array
    :return: fwhm value in pixels
    """

    assert len(xi) == len(data), 'Input arrays are not the same dimension'

    if interp:
        # interpolate to larger size
        x = np.linspace(xi[0], xi[-1])
        dat = np.interp(x, xi, data)
    else:
        x = xi
        dat = data

    # obtain values
    mean = np.mean(dat)
    # fit curve to gaussian
    popt, _ = optimize.curve_fit(gauss, x, dat, p0=[1, mean, np.std(dat)])
    # get half max x locations
    # try:
    #     r1, r2 = __get_FWHM(x, gauss(x, *popt))
    # except ValueError:
    #     print('Spline fit for FWHM failed. Check your data.')
    #     return float('NaN')
    # except:
    #     print('Something has gone wrong. Check your data')
    #     return float('NaN')
    # else:
    # indent me here
    # get FWHM
    #fwhm = r2 - r1
    # data has been normalized from 0-255 to 0-1, hence multiplying by 255 to scale back up
    fwhm = 2.355*np.std(dat)*255
    # plot
    if plot:
        # print('X1: %f, X2: %f, \nFWHM: %f Pixels' % (r1, r2, fwhm))
        plt.plot(x, gauss(x, *popt), label='Gaussian Fit')
        plt.plot(x, dat, label='Data')
        plt.xlabel('Pixel (arb)')
        plt.ylabel('Normalized Integrated Pixel Value')
        plt.axvspan(popt[1] - 0.5*fwhm, popt[1] + 0.5*fwhm, color='y', alpha=0.25, lw=0, label='FWHM')
        plt.legend()
        plt.show()

    return fwhm, popt


def gauss(x, a, x0, sigma):
    # return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    return a * np.exp(-((x - x0) / 4 / sigma) ** 2)


def __get_FWHM(x, data):
    # data should be normalized to [0, 1]
    spline = UnivariateSpline(x, data - max(data) / 2, s=0)  # removed s=0 as parameter
    # find roots of half max locations
    r1, r2 = spline.roots()
    return r1, r2


if __name__ == '__main__':
    pxpitch = 4.8   # micron
    dat = np.asarray(Image.open('Examples/FLIR2.bmp'))

    st = time.time()
    try:
        (fwhmx, fwhmy), (xopt, yopt) = get_fwhm(dat, rfactor=1, plot=False)

        x = np.linspace(0, len(dat[0]), len(dat[0]))
        y = np.linspace(0, len(dat), len(dat))

        # xx = gauss(x, *xopt)
        # yy = gauss(y, *yopt)
        # z = np.zeros((xx.shape[0], yy.shape[0]))
        # for i in range(len(xx) - 1):
        #     for j in range(len(yy) - 1):
        #         z[i][j] = xx[i]+yy[j]
        #
        #
        # z = z.transpose()
        # X, Y = np.meshgrid(x, y)
        #
        # print(X.shape, Y.shape, z.shape)
        #
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # #ax.plot_wireframe(x, y, dat)
        # print(X.shape, Y.shape, z.shape)
        # surf = ax.plot_surface(X, Y, z, cmap='Spectral')
        # plt.show()

    except Exception as e:
        print(e)
        print('Could not fit data')
        plt.text(0, -65, 'FWHM Values could not be obtained')
    else:

        print(xopt, yopt)
        meanval = np.mean(dat)
        print('Took ' + str(time.time() - st) + ' seconds.')
        print("fwhm x:", fwhmx, "->", fwhmx * pxpitch, 'micron')
        print("fwhm y:", fwhmy, "->", fwhmy * pxpitch, 'micron')
        print('Mean pixel value:', str(meanval))
        print('\n')
        plt.plot([xopt[1] - 0.5 * fwhmx, xopt[1] + 0.5 * fwhmx], [yopt[1], yopt[1]], c='black')
        plt.plot([xopt[1], xopt[1]], [yopt[1] - 0.5 * fwhmy, yopt[1] + 0.5 * fwhmy], c='black')
        plt.scatter([xopt[1]], [yopt[1]], c='black')
        plt.text(0, -65,
                 str("FWHM x: " + str(fwhmx)[:6] + " Pixels -> " + str(fwhmx * pxpitch)[:6] + ' micron '))
        plt.text(0, -15,
                 str("FWHM y: " + str(fwhmy)[:6] + " Pixels -> " + str(fwhmy * pxpitch)[:6] + ' micron '))

    plt.imshow(dat, cmap='Spectral')
    plt.xlabel('Pixel Number')
    plt.ylabel('Pixel Number')
    plt.colorbar()

    plt.show()
