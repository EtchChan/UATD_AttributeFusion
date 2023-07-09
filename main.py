# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from scipy.integrate import quad
import math

import Fit_Skewed_Gaussian_Model as fsgm


def parse_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    sonar = root.find('sonar')
    range_val = sonar.find('range').text
    frequency_val = sonar.find('frequency').text.replace('k', '')  # remove the 'k' from the frequency value
    return range_val, frequency_val


# Function to swap values if the first is greater than the second
def swap_value(a, b):
    if a > b:
        return b, a
    return a, b


# Parse the xml files and get bndbox values
def parse_xml_bndbox(inputFolder, outputFolder):
    # dictionary that map object name to integer
    class_map = {
        'ball': 0,
        'circle cage': 1,
        'cube': 2,
        'cylinder': 3,
        'human body': 4,
        'metal bucket': 5,
        'plane': 6,
        'rov': 7,
        'square cage': 8,
        'tyre': 9,
    }

    file_list = sorted(glob.glob(os.path.join(inputFolder, '*.xml')))
    cnt = 0
    for file in file_list:
        cnt += 1
        tree = ET.parse(file)
        root = tree.getroot()

        # Extract size and elevation information
        width = root.find("./size/width").text
        height = root.find("./size/height").text
        elevation = root.find("./sonar/elevation").text

        # Initialize the output content with size and elevation information
        content = f"{width} {height} {elevation}\n"

        # Iterate through all objects
        for obj in root.findall("./object"):
            name = class_map[obj.find("name").text]
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            # check for zero area anomaly
            if xmin == xmax or ymin == ymax:
                continue
            # check and revise negative area anomaly
            xmin, xmax = swap_value(xmin, xmax)
            ymin, ymax = swap_value(ymin, ymax)
            # check for out of range anomaly
            if xmin < 0 or ymin < 0:
                continue
            elif xmax >= int(height):
                continue
            elif xmax >= int(width) > ymax:
                xmin, ymin = swap_value(xmin, ymin)
                xmax, ymax = swap_value(xmax, ymax)

            # Append the object information to the output content
            content += f"{name} {xmin} {ymin} {xmax} {ymax}\n"

        # Write the output content to corresponding txt file
        with open(os.path.join(outputFolder, os.path.basename(file).replace('.xml', '.txt')), 'w') as f:
            f.write(content)


# function the plot data pairs in txt file
def plot_data_pairs(file_path):
    distance = []
    intensity = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.split()
            distance.append(float(data[0]))
            intensity.append(float(data[1]))

    # create figure
    plt.figure(figsize=(10, 6))
    plt.plot(distance, intensity, marker='o')

    # specify the title and labels
    plt.title('720kHz: Attenuation ratio vs. Range')
    plt.xlabel('Range (m)')
    plt.ylabel('Attenuation ratio')

    # save the plot
    plt.savefig('D:/Repo/UATD/Dataset/Training/Attenuation_data_720.png')

    # show the plot
    plt.show()


def erf_integrand(t):
    return math.exp(-(t ** 2) / 2.0)


def erf(x):
    return (2.0 / math.sqrt(math.pi * 2.0)) * quad(erf_integrand, 0, x)[0]


# Function to draw bounding boxes on an image
def draw_boxes(image, boxes, line_width=3, color='white'):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(((box[1], box[2]), (box[3], box[4])), outline=color, width=line_width)
    return image


# Function to draw histogram of given data
def draw_histogram(inputFile):
    # load the data from txt file
    data = np.loadtxt(inputFile, dtype=float)

    # Separate the data into three groups
    area_average = data[:, 0]
    row_average = data[:, 1]
    ratio = data[:, 2]

    # specify the number of bins
    bins = 60
    # create histograms
    area_hist, area_bin_edges = np.histogram(area_average, bins=bins)
    row_hist, row_bin_edges = np.histogram(row_average, bins=bins)
    ratio_hist, ratio_bin_edges = np.histogram(ratio, bins=bins)

    # Compute the normalized frequency values and mean values for each bin
    area_freq = area_hist / np.sum(area_hist)
    area_mean = 0.5 * (area_bin_edges[:-1] + area_bin_edges[1:])
    row_freq = row_hist / np.sum(row_hist)
    row_mean = 0.5 * (row_bin_edges[:-1] + row_bin_edges[1:])
    ratio_freq = ratio_hist / np.sum(ratio_hist)
    ratio_mean = 0.5 * (ratio_bin_edges[:-1] + ratio_bin_edges[1:])

    # Get the directory name and replace the basename with the output filenames
    area_file = inputFile.replace(".txt", "_area_avg_hist.txt")
    row_file = inputFile.replace(".txt", "_row_avg_hist.txt")
    ratio_file = inputFile.replace(".txt", "_ratio_hist.txt")

    # Save the results to text files
    np.savetxt(area_file, np.vstack((area_mean, area_freq)).T)
    np.savetxt(row_file, np.vstack((row_mean, row_freq)).T)
    np.savetxt(ratio_file, np.vstack((ratio_mean, ratio_freq)).T)

    # create histograms
    plt.figure(figsize=(18, 6))
    plt.suptitle('1200kHz: histogram of statistical measures', fontsize=16, fontweight='bold')

    plt.subplot(1, 3, 1)
    plt.hist(area_average, bins=bins, color='blue', edgecolor='black', alpha=0.5)
    plt.title('Target-area-average')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(row_average, bins=bins, color='blue', edgecolor='black', alpha=0.5)
    plt.title('Target-row-average')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(ratio, bins=bins, color='blue', edgecolor='black', alpha=0.5)
    plt.title('Ratio of area average to row average')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')

    # save the plot
    plt.savefig(inputFile.replace('.txt', '_hist.png'))

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ********* extract the range and frequency values from the xml files *********#
    # file_list = sorted(glob.glob(os.path.join('D:/Repo/UATD/Dataset/Test/annotations', '*.xml')))
    # with open('D:/Repo/UATD/Dataset/Test/range_frequency.txt', 'w') as f:
    #     for file in file_list:
    #         range_val, frequency_val = parse_xml(file)
    #         f.write(range_val + ' ' + frequency_val + '\n')

    # plot the extracted range-frequency data pairs
    # plot_data_pairs('D:/Repo/UATD/Dataset/Training/Attenuation_data_720.txt')

    # parse the xml files and get bndbox values
    # parse_xml_bndbox('D:/Repo/UATD/Dataset/Test/annotations',
    #                  'D:/Repo/UATD/Dataset/Processed/Thorp/Test/annotations')

    # calculate and save the erf value of skewed normal distribution
    # with open("D:/Repo/UATD/Dataset/Processed/Thorp/Train/erf_lookup_table.txt", 'w') as f:
    #     for x in np.arange(0, 4.01, 0.01):
    #         result = erf(x)
    #         f.write(f"{x} {result}\n")

    # ********* display the filtered image and original one in same figure to compare them *********#
    # origin_dir = "D:/Repo/UATD/Dataset/Training/images/"
    # thorp_dir = "D:/Repo/UATD/Dataset/Processed/Thorp/Train/images/"
    # guassian_dir = "D:/Repo/UATD/Dataset/Processed/Gaussian/Train/images/"
    # annotation_dir = "D:/Repo/UATD/Dataset/Processed/Thorp/Train/annotations/"
    # display_dir = "D:/Repo/UATD/Dataset/Processed/Display/"
    # # Create the display directory if it doesn't exist
    # if not os.path.exists(display_dir):
    #     os.makedirs(display_dir)
    # # Sampling over images
    # for i in range(2, 7601, 600):
    #     # Load images
    #     origin_img = Image.open(os.path.join(origin_dir, f"{str(i).zfill(5)}.bmp"))
    #     thorp_img = Image.open(os.path.join(thorp_dir, f"{str(i).zfill(5)}.bmp"))
    #     guassian_img = Image.open(os.path.join(guassian_dir, f"{str(i).zfill(5)}.bmp"))
    #     # Load annotations of bounding boxes
    #     with open(os.path.join(annotation_dir, f"{str(i).zfill(5)}.txt"), 'r') as file:
    #         boxes = [[int(num) for num in line.split()] for line in file.readlines()[1:]]
    #     # Draw bounding boxes on images
    #     origin_img = draw_boxes(origin_img, boxes)
    #     thorp_img = draw_boxes(thorp_img, boxes)
    #     guassian_img = draw_boxes(guassian_img, boxes)
    #     # Create a new figure with 3 subplots
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #     # Display the images
    #     ax1.imshow(origin_img, cmap='gray')
    #     ax2.imshow(thorp_img, cmap='gray')
    #     ax3.imshow(guassian_img, cmap='gray')
    #     # save the plot
    #     plt.savefig(os.path.join(display_dir, f"{str(i).zfill(5)}.png"))
    #     plt.close(fig)

    # ********* display and save only one group of specified images with boundbox as example *********#
    img = Image.open("D:/Repo/UATD/Dataset/UATD_Training/UATD_Training/images/07581.bmp")
    img_thorp = Image.open("D:/Repo/UATD/Dataset/Processed/Thorp/Train/images/07581.bmp")
    img_gaussian = Image.open("D:/Repo/UATD/Dataset/Processed/Gaussian/Train/images/07581.bmp")
    img_fusion = Image.open("D:/Repo/UATD/Dataset/Processed/Fusion/Train/images/07581.bmp")

    with open("D:/Repo/UATD/Dataset/Processed/Thorp/Train/annotations/07581.txt", 'r') as file:
        boxes = [[int(num) for num in line.split()] for line in file.readlines()[1:]]

    # Draw bounding boxes on images
    img = draw_boxes(img, boxes, 6)
    img_thorp = draw_boxes(img_thorp, boxes, 6)
    img_gaussian = draw_boxes(img_gaussian, boxes, 6)
    img_fusion = draw_boxes(img_fusion, boxes, 6)

    # Create a new figure with 4 subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    captions = ['(c1) Raw image', '(c2) Enhanced by attenuation model', '(c3) Enhanced by probability model', '(c4) Fusion result']

    for idx, (ax, img, caption) in enumerate(zip(axs, [img, img_thorp, img_gaussian, img_fusion], captions)):
        if idx == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        # set the title at the bottom of the figure
        ax.set_title(caption, y=-0.15)

    # save the plot
    # plt.tight_layout()
    plt.savefig("D:/Repo/UATD/Dataset/Processed/Display/eg_07581_result.png")
    plt.show()

    # ******** show the histograms of target area average intensities ********#
    # draw_histogram("D:/Repo/UATD/Dataset/Training/target_average_720.txt")
    # draw_histogram("D:/Repo/UATD/Dataset/Training/target_average_1200.txt")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
