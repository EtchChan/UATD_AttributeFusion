import os
import xml.etree.ElementTree as ET



def remove_unit_and_convert_to_float(input_string):
    # Remove non-digit characters from the input string
    cleaned_string = ''.join(char for char in input_string if char.isdigit() or char == '.')

    # Convert the cleaned string to a float
    float_number = float(cleaned_string)

    return float_number


def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = {}

    # Parse sonar data
    sonar = root.find('sonar')
    data['sonar'] = {
        'range': float(sonar.find('range').text),
        'azimuth': int(sonar.find('azimuth').text),
        'elevation': int(sonar.find('elevation').text),
        'soundspeed': float(sonar.find('soundspeed').text),
        'frequency': remove_unit_and_convert_to_float(sonar.find('frequency').text)
    }

    # Parse file data
    file = root.find('file')
    data['file'] = {
        'folder': file.find('folder').text,
        'filename': file.find('filename').text
    }

    # Parse size data
    size = root.find('size')
    data['size'] = {
        'width': int(size.find('width').text),
        'height': int(size.find('height').text),
        'channel': int(size.find('channel').text)
    }

    # Parse object data
    obj = root.find('object')
    bndbox = obj.find('bndbox')
    data['object'] = {
        'name': obj.find('name').text,
        'bndbox': {
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        }
    }

    return data


def process_files_in_batch(folder_path, start, end):
    all_data = {}
    for i in range(start, end + 1):
        file_name = f"{i:05d}.xml"  # Format the file name with leading zeros
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            parsed_data = parse_xml(file_path)
            all_data[file_name] = parsed_data
        else:
            print(f"File {file_path} not found")

    return all_data


if __name__ == "__main__":
    # Batch Task Version

    # folder_path = "path/to/your/folder"
    # start = 1
    # end = 10000
    # all_parsed_data = process_files_in_batch(folder_path, start, end)
    # print(all_parsed_data)

    # Single File Version

    file_path = "D:/Repo/UATD/Dataset/00002.xml"
    parsed_data = parse_xml(file_path)
    print(parsed_data)
