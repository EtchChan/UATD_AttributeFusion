from PIL import Image, ImageEnhance, ImageOps
from XML_parse import parse_xml
import numpy as np
import cv2
import math


# restore the reflectivity of objects according to the thorp model
def thorp_model(sonar_image, parameters):
    # get the size of the image
    azimuth, length = sonar_image.size

    # get the pixel data
    pixel_data = sonar_image.load()

    # convert the image to numpy array
    im_array = np.array(sonar_image)

    # Normalize pixel values to the range [0, 1] for each channel
    im_array_norm = np.zeros_like(im_array, dtype=np.float32)
    for i in range(im_array.shape[2]):
        im_array_norm[:, :, i] = im_array[:, :, i].astype(np.float32) / 255.0

    # get the scan range
    scan_range = parameters['sonar']['range']

    # get the frequency
    frequency = parameters['sonar']['frequency']

    # calculate the absorption coefficient
    absorption_coefficient = 0.1 * frequency ** 2 / (1 + frequency ** 2) \
                             + 40 * frequency ** 2 / (4100 + frequency ** 2) \
                             + 2.75 * 10 ** -4 * frequency ** 2 + 0.003
    absorption_coefficient = float(absorption_coefficient)

    # apply the thorp model
    for y in range(azimuth):
        for x in range(length):
            # get the pixel value
            pixel_value = im_array_norm[x, y]

            # calculate the distance
            distance = (x * scan_range / length + 0.1) / 1000.0

            # calculate the attenuation coefficient
            absorption_part = 10 ** (absorption_coefficient * distance / 10.0)
            distance_part = distance ** 1.5
            # distance_part should be guaranteed to be greater than 1
            if distance_part < 1.0:
                distance_part = 1.0
            attenuation_coefficient = distance_part * absorption_part
            # log_attenuation_coefficient = 1.5 * math.log10(distance) + absorption_coefficient * distance / 10.0

            # calculate the reflectivity
            reflectivity = tuple(elem * attenuation_coefficient ** 2 for elem in pixel_value)

            # using reflectivity as pixel value
            pixel_value = reflectivity

            # set the pixel value
            im_array_norm[x, y] = pixel_value

    # normalize the image
    max_value = np.max(im_array_norm)
    min_value = np.min(im_array_norm)
    im_array_norm = (im_array_norm - min_value) / (max_value - min_value)

    # convert the image back to PIL image
    processed_image = Image.fromarray((im_array_norm * 255).astype(np.uint8))
    return processed_image


def gamma_correction(image, gamma):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    lut = np.array([(1.0 - (i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")

    # Apply gamma correction to each channel using the lookup table
    return Image.fromarray(cv2.LUT(np.array(image), lut))


# mark the target in the image according to the xml file
def mark_target(image, paramters):
    # get boundbox from the xml file
    x_min = paramters['object']['bndbox']['xmin']
    y_min = paramters['object']['bndbox']['ymin']
    x_max = paramters['object']['bndbox']['xmax']
    y_max = paramters['object']['bndbox']['ymax']

    # correct the order of the annotation
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    # get the pixel data
    pixel_data = image.load()

    # set the color of the boundbox
    color = (255, 0, 0)

    # draw the boundbox
    for x in range(x_min, x_max):
        pixel_data[x, y_min] = color
        pixel_data[x, y_max] = color
    for y in range(y_min, y_max):
        pixel_data[x_min, y] = color
        pixel_data[x_max, y] = color

    return image


def enhance_contrast(image_path, output_path, gamma):
    # Open the image
    image = Image.open(image_path)

    # get parameters from the xml file
    xml_path = image_path.replace(".bmp", ".xml")
    parameters = parse_xml(xml_path)

    # Apply thorp model
    sonar_image = thorp_model(image, parameters)

    # Apply gamma correction
    corrected_image = gamma_correction(sonar_image, gamma)

    # Save the result
    corrected_image.save(output_path)

    # Mark the target
    marked_origin_image = mark_target(image, parameters)
    marked_corrected_image = mark_target(corrected_image, parameters)
    marked_origin_image.save(image_path.replace(".bmp", "_marked.bmp"))
    marked_corrected_image.save(output_path.replace(".bmp", "_marked.bmp"))


if __name__ == "__main__":
    input_image_path = "D:/Repo/UATD/Dataset/00003.bmp"
    output_image_path = "D:/Repo/UATD/Dataset/00003_output.bmp"
    gamma_value = 3.10  # Adjust this value to your desired level of gamma correction

    enhance_contrast(input_image_path, output_image_path, gamma_value)
