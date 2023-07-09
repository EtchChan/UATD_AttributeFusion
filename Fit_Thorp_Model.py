import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def model_func(d, alpha, k):
    return np.exp(-alpha * d) * (d / 0.1 + 1) ** (-k)


def fit_thorp_model():
    # calculate the starting absorption coefficient using theoretical model
    frequency = 1200.0
    absorption_coefficient = 0.1 * frequency ** 2 / (1 + frequency ** 2) \
                             + 40 * frequency ** 2 / (4100 + frequency ** 2) \
                             + 2.75 * 10 ** -4 * frequency ** 2 + 0.003
    absorption_coefficient = float(absorption_coefficient) / 1000.0 * np.log(10) / 3.0
    k = 0.09

    # load the data
    distance_values = []
    intensity_values = []

    # read the data from text file
    with open('D:/Repo/UATD/Dataset/Training/outlier_removed_720.txt', 'r') as f:
        for line in f:
            data = line.split()
            distance_values.append(float(data[0]))
            intensity_values.append(float(data[1]))

    # convert the lists to numpy arrays
    distance_values = np.array(distance_values)
    intensity_values = np.array(intensity_values)

    # convert the 0 from nan to number
    distance_values = np.nan_to_num(distance_values)
    intensity_values = np.nan_to_num(intensity_values)

    # using curve_fit to find the best parameters
    popt, pcov = curve_fit(model_func, distance_values, intensity_values, p0=[absorption_coefficient, k])

    # popt contains the best fit parameters, print them out
    print("alpha = %f, k = %f" % (popt[0], popt[1]))

    # plot the original data
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, intensity_values, 'b-', label='data')

    # plot the fitted curve
    fit_values = model_func(distance_values, *popt)
    plt.plot(distance_values, fit_values, 'ro', label='Fit: alpha=%5.3f, k=%5.3f' % tuple(popt))

    # set the labels
    plt.title('720kHz: Attenuation Model Fitting')
    plt.xlabel('Distance (m)')
    plt.ylabel('Attenuation ratio')
    plt.legend()

    # save and display the plot
    plt.savefig('D:/Repo/UATD/Dataset/Training/Thorp_Fit_720.png')
    plt.show()


if __name__ == '__main__':
    fit_thorp_model()
