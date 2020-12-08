import os
import math
import codecs
import numpy as np
from PIL import Image, ImageEnhance
from config import train_parameters


def resize_img(img, target_size):
    """
    Force scale image
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img


def random_crop(img, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((train_parameters['input_size'][1], train_parameters['input_size'][2]), Image.BILINEAR)
    return img


def rotate_image(img):
    """
    Image enhancement, increase random rotation Angle
    """
    angle = np.random.randint(-14, 15)
    img = img.rotate(angle)
    return img


def random_brightness(img):
    """
    Image enhancement, brightness adjustment
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_enhance_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    """
    Image enhancement, contrast adjustment
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_enhance_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    """
    Image enhancement, saturation adjustment
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_enhance_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    """
    Image enhancement, color adjustment
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['hue_prob']:
        hue_delta = train_parameters['image_enhance_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_color(img):
    """
    Image enhancement of probability
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob < 0.35:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    elif prob < 0.7:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def custom_image_reader(file_list, data_dir, mode):
    """
    customize reader
    initialize image type and number
    :param file_list:
    :param data_dir:
    :param mode:
    :return:
    """
    with codecs.open(file_list) as flist:
        lines = [line.strip() for line in flist]

    def reader():
        np.random.shuffle(lines)
        for line in lines:
            if mode == 'train':
                img_path, label = line.split()
                img = Image.open(img_path)
                try:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if train_parameters['image_enhance_strategy']['need_distort']:
                        img = distort_color(img)
                    if train_parameters['image_enhance_strategy']['need_rotate']:
                        img = rotate_image(img)
                    if train_parameters['image_enhance_strategy']['need_crop']:
                        img = random_crop(img, train_parameters['input_size'])
                    if train_parameters['image_enhance_strategy']['need_flip']:
                        mirror = int(np.random.uniform(0, 2))
                        if mirror == 1:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    # HWC--->CHW && normalized
                    img = np.array(img).astype('float32')
                    img -= train_parameters['mean_rgb']
                    img = img.transpose((2, 0, 1))  # HWC to CHW
                    img *= 0.007843                 # Normalized pixel values
                    yield img, int(label)
                except Exception as e:
                    pass
            if mode == 'val':
                img_path, label = line.split()
                img = Image.open(img_path)

                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = resize_img(img, train_parameters['input_size'])
                # HWC--->CHW && normalized
                img = np.array(img).astype('float32')
                img -= train_parameters['mean_rgb']
                img = img.transpose((2, 0, 1))
                img *= 0.007843
                yield img, int(label)
            elif mode == 'test':
                img_path, _ = line.split()
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = resize_img(img, train_parameters['input_size'])
                # HWC--->CHW && normalized
                img = np.array(img).astype('float32')
                img -= train_parameters['mean_rgb']
                img = img.transpose((2, 0, 1))
                img *= 0.007843
                yield img

    return reader
