import numpy as np
from PIL import Image
def RegDB_data():
    data_dir = './datasets/RegDB/'
    train_color_list = data_dir + 'idx/train_visible_1' + '.txt'
    train_thermal_list = data_dir + 'idx/train_thermal_1' + '.txt'

    color_img_file, train_color_label = load_data(train_color_list)
    train_rgb_label = np.array(train_color_label)
    thermal_img_file, train_thermal_label = load_data(train_thermal_list)
    train_ir_label = np.array(train_thermal_label)

    train_color_image = []
    for i in range(len(color_img_file)):
        img = Image.open(data_dir + color_img_file[i])
        img = img.resize((144, 288), Image.ANTIALIAS)
        pix_array = np.array(img)
        train_color_image.append(pix_array)
    train_color_image = np.array(train_color_image)
    np.save("./RegDB_train_rgb.npy", train_color_image)
    np.save("./RegDB_train_rgb_label.npy", train_rgb_label)


    train_thermal_image = []
    for i in range(len(thermal_img_file)):
        img = Image.open(data_dir + thermal_img_file[i])
        img = img.resize((144, 288), Image.ANTIALIAS)
        pix_array = np.array(img)
        train_thermal_image.append(pix_array)
    train_thermal_image = np.array(train_thermal_image)
    np.save("./RegDB_train_ir.npy", train_thermal_image)
    np.save("./RegDB_train_ir_label.npy", train_ir_label)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
    return file_image, file_label


if __name__=="__main__":
    RegDB_data()