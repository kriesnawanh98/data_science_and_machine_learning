def downgrading_crop_image(character="0", file_name="handyangka_03.jpg"):
    import numpy as np
    import cv2
    import tensorflow as tf

    path = f"./dataset/test/{character}/{file_name}"
    img = cv2.imread(path, 1)
    print("Original Shape = ", img.shape)

    width_ori_size = img.shape[1]  #194
    height_ori_size = img.shape[0]  #208

    width_crop_left = int((width_ori_size - 150) / 2)
    width_crop_right = int(width_ori_size - (width_ori_size - 150) / 2)

    height_crop_top = int((height_ori_size - 150) / 2)
    height_crop_bottom = int(height_ori_size - (height_ori_size - 150) / 2)
    print(height_crop_bottom)
    img_crop = img[width_crop_left:width_crop_right,
                   height_crop_top:height_crop_bottom]

    img_downgraded = cv2.resize(img_crop, (0, 0), fx=28 / 150, fy=28 / 150)
    image_invert = cv2.bitwise_not(img_downgraded)

    numpy_array = np.array(image_invert)
    numpy_array.resize(28, 28)

    numpy_array_norm = numpy_array / 255.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    numpy_array_3d = numpy_array_norm[:, :, np.newaxis]
    numpy_array_4d = np.array([numpy_array_3d])
    return numpy_array_4d


if __name__ == '__main__':
    x = downgrading_crop_image()
    print(x.shape)