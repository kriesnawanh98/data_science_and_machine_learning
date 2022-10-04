def convert_image_to_array(path="./dataset/train/benign/",
                           label="benign/malignant"):
    from PIL import ImageOps as io
    import numpy as np
    import os
    from PIL import Image

    X_train = []
    for file_name in os.listdir(path):
        data_img = Image.open(f"{path}{file_name}")
        # greyscale_data = io.grayscale(data_img)  #change from RGB to greyscale
        inv_data = io.invert(data_img)  #invert data/flip
        numpy_array = np.array(inv_data.getdata())
        numpy_array.resize(224, 224, 3)
        X_train.append(numpy_array)

    if label == "benign":
        y_train = np.zeros(np.array(X_train).shape[0])
    else:
        y_train = np.ones(np.array(X_train).shape[0])

    return np.array(X_train), y_train


if __name__ == '__main__':
    X_train, y_train = convert_image_to_array(label="benign")
    print(X_train.shape)
    print(y_train.shape)