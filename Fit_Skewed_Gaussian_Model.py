import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# global variables
table = np.loadtxt('D:/Repo/UATD/Dataset/Processed/Thorp/Train/erf_lookup_table.txt')  # Load the look-up table
x_table = table[:, 0]
y_table = table[:, 1]


# Define the function for normalizing the data
def normalize(x, mu, sigma):
    return (x - mu) / sigma


# Define the Normal distribution function
def normal(std_x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-std_x ** 2 / 2) * ((np.max(std_x) - np.min(std_x)) / len(std_x))


# Define the skewness function using the look-up table
def skewness(std_x, alpha):
    sign = np.sign(std_x)
    return np.where(abs(alpha * std_x) <= 4, 1 + sign * np.interp(alpha * std_x, x_table, y_table), 1 + sign * 1)


# Define the skewed Gaussian function
def skewed_gaussian(x, mu, sigma, alpha):
    return normal(normalize(x, mu, sigma)) * skewness(normalize(x, mu, sigma), alpha)


# Define the function to fit the skewed Gaussian function
def fit_skewed_gaussian(inputFolder):
    # Load the mean and stddev data
    mean_stddev = np.loadtxt(inputFolder + 'mean_stddev.txt')

    # Create the figures and axes
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle("720kHz: skewed normal model fitting", fontsize=16, fontweight='bold')
    fig2.suptitle("1200kHz: skewed normal model fitting", fontsize=16, fontweight='bold')

    axes = [axes1, axes2]

    # Set up the titles mapping
    titles_map = {
        0: "target-area-average",
        1: "target-row-average",
        2: "ratio",
    }

    # Load and fit each data set
    for i, filename in enumerate([inputFolder + 'target_average_720_area_avg_hist.txt',
                                  inputFolder + 'target_average_720_row_avg_hist.txt',
                                  inputFolder + 'target_average_720_ratio_hist.txt',
                                  inputFolder + 'target_average_1200_area_avg_hist.txt',
                                  inputFolder + 'target_average_1200_row_avg_hist.txt',
                                  inputFolder + 'target_average_1200_ratio_hist.txt']):
        # Load the data
        data = np.loadtxt(filename)
        x = data[:, 0]
        y = data[:, 1]

        # Fit the data with the skewed Gaussian function
        p0 = [mean_stddev[i, 0], mean_stddev[i, 1], 2.4]
        popt, pcov = curve_fit(skewed_gaussian, x, y, p0=p0)

        # Print the best-fit alpha value
        print(f'{filename}: mu=%e, sigma=%e, alpha=%e \n' % tuple(popt))

        # Select the correct axis
        ax = axes[i // 3][i % 3]

        # Plot the data and the fitted curve
        fitted_y = skewed_gaussian(x, *popt)
        ax.plot(x, y, 'b-', label='data')
        ax.plot(x, fitted_y, 'r-', label='Fit: mu=%5.3f, sigma=%5.3f, alpha=%5.3f' % tuple(abs(popt)))
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("normalized frequency")
        ax.set_title(titles_map[i % 3])

    # Save the figures
    fig1.savefig(inputFolder + "figure_720.png")
    fig2.savefig(inputFolder + "figure_1200.png")

    plt.show()


# Run the function
if __name__ == '__main__':
    fit_skewed_gaussian('D:/Repo/UATD/Dataset/Training/')
