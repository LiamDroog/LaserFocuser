from FWHMFinder import get_fwhm, gauss
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import tkinter as tk
from BlackFlyCameraClass import RunBlackFlyCamera
from scipy.optimize import curve_fit
from droogCNC import TwoAxisStage

class FocusingHelper:
    # todo: logic to sweep across
    def __init__(self):

        self.position_dict = {}
        self.bounds = [-10, 10]
        self.steps = 10
        self.positions = np.linspace(self.bounds[0], self.bounds[1], self.steps)
        self.stage = TwoAxisStage('COM4', '115200', 'Config/startup.txt')
        self.window = tk.Tk(className='Focus Assist')
        self.window.geometry(
            '%dx%d+%d+%d' % (int(self.window.winfo_screenwidth() * 0.15), int(self.window.winfo_screenheight() * 0.15),
                             self.window.winfo_screenwidth() / 4,
                             self.window.winfo_screenheight() / 5))
        self.window.rowconfigure([i for i in range(5)], minsize=25, weight=1)
        self.window.columnconfigure([i for i in range(5)], minsize=25, weight=1)
        self.button = tk.Button(master=self.window, text='Capture Image', command=self.takeimage)
        self.button.grid(row=0, column=0, rowspan=5, columnspan=2, sticky='nsew')
        self.sweepbutton = tk.Button(master=self.window, text='Sweep focus', command=self.sweepFocusX)
        self.sweepbutton.grid(row=2, column=0, sticky='nsew')
        self.exposureinput = tk.Entry(master=self.window)
        self.exposureinput.grid(row=0, column=2, columnspan=2)
        self.setexposure = tk.Button(master=self.window, text='Set Exposure Time (us)',
                                     command=lambda: self.changeparam('ExposureTime', self.exposureinput,
                                                                      self.setexposure))
        self.setexposure.grid(row=0, column=5, sticky='nesw')
        self.bfs = RunBlackFlyCamera('19129388', 0)
        self.bfs.adjust('GainAuto', 'Off')
        self.bfs.adjust('ExposureAuto', 'Off')
        self.bfs.adjust('ExposureTime', 6)
        self.window.protocol("WM_DELETE_WINDOW", self.__on_closing)
        self.pxpitch = 4.8
        self.window.mainloop()


    def takeimage(self):
        self.bfs.start()
        im = self.bfs.get_image_array()
        (fwhmx, fwhmy), (xopt, yopt) = get_fwhm(im, rfactor=1, plot=False)
        #self.analyze(im)
        self.bfs.stop()
        print('FWHM: ', 0.5*(fwhmx+fwhmy))
        return 0.5*(fwhmx+fwhmy)

    def changeparam(self, target, inputbox, button):
        try:
            val = inputbox.get()
            txt = button['text']
            self.bfs.adjust(target, val)
        except:
            button['text'] = 'Failed to configure output'
        else:
            print('Succeeded')
        finally:
            self.window.after(4000, lambda: button.configure(text=txt))

    def sweepFocusX(self):
        # see misc test before implement
        self.stage.sendCommand('G90')
        self.goTo(self.bounds[0])
        for i in self.positions:
            print('Going to %f' % i)
            self.goTo(i)
            self.position_dict[i] = self.takeimage()

        px = self.findFocalPoint()
        self.goTo(px)

    def goTo(self, point):
        self.stage.sendCommand('G0 X%2.3f' % point)
        time.sleep(1)


    def findFocalPoint(self):
        x = []
        y = []
        for key, val in sorted(self.position_dict.items()):
            print(key, val)
            x.append(key)
            y.append(val)
        xi = np.linspace(x[0], x[-1])
        yi = np.interp(xi, x, y)

        popt, _ = curve_fit(self.parabola, xi, yi)
        print('vertex: (%f, %f)' % (popt[1], popt[2]))
        plt.plot(xi, self.parabola(xi, *popt))
        plt.scatter(x, y)
        plt.show()
        return popt[1]

    def parabola(self, x, a, b, c):
        return (a * (x - b) ** 2) + c

    def analyze(self, dat, plot=False):
        plt.clf()
        st = time.time()
        try:
            (fwhmx, fwhmy), (xopt, yopt) = get_fwhm(dat, rfactor=1, plot=False)
        except:
            print('Could not fit data')
            plt.text(0, -65, 'FWHM Values could not be obtained')
        else:

            meanval = np.mean(dat)
            print('Took ' + str(time.time() - st) + ' seconds.')
            print("fwhm x:", fwhmx, "->", fwhmx*self.pxpitch, 'micron')
            print("fwhm y:", fwhmy, "->", fwhmy*self.pxpitch, 'micron')
            print('Mean pixel value:', str(meanval))
            print('\n')
            plt.plot([xopt[1] - 0.5 * fwhmx, xopt[1] + 0.5 * fwhmx], [yopt[1], yopt[1]], c='black')
            plt.plot([xopt[1], xopt[1]], [yopt[1] - 0.5 * fwhmy, yopt[1] + 0.5 * fwhmy], c='black')
            plt.scatter([xopt[1]], [yopt[1]], c='black')
            plt.text(0, -65,
                     str("FWHM x: " + str(fwhmx)[:6] + " Pixels -> " + str(fwhmx * self.pxpitch)[:6] + ' micron '))
            plt.text(0, -15,
                     str("FWHM y: " + str(fwhmy)[:6] + " Pixels -> " + str(fwhmy * self.pxpitch)[:6] + ' micron '))

        plt.imshow(dat, cmap='Spectral')
        plt.xlabel('Pixel Number')
        plt.ylabel('Pixel Number')
        plt.colorbar()
        plt.show()

    def __on_closing(self):
        self.bfs.close()
        self.window.destroy()


if __name__ == '__main__':
    FocusingHelper()
