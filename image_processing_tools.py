# This module defines functions, which are used in the implementation of the data
# augmentation scheme

import numpy as np
import cv2 as cv
import math


# this function rotates an image without cropping (angle in degrees)
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    # function getRotationMatrix2D uses coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


# this function applies shear mapping to image
def shear_image(img, lambda1, lambda2):
    t = np.float32([[1, lambda2, 0],
                    [lambda1, 1, 0],
                    [0, 0, 1]])
    trans = t[0:2, :]

    inv_t = np.linalg.inv(t)
    inv_trans = inv_t[0:2, :]

    # get the sizes
    h, w = img.shape[:2]

    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst_pts = cv.transform(np.array([src_pts]), trans)[0]

    min_x, max_x = np.min(dst_pts[:, 0]), np.max(dst_pts[:, 0])
    min_y, max_y = np.min(dst_pts[:, 1]), np.max(dst_pts[:, 1])

    dst_w = int(max_x - min_x + 1)
    dst_h = int(max_y - min_y + 1)

    dst_center = np.float32([[(dst_w - 1.0) / 2, (dst_h - 1.0) / 2]])
    src_projected_center = cv.transform(np.array([dst_center]), inv_trans)[0]

    translation = src_projected_center - np.float32([[(w - 1.0) / 2, (h - 1.0) / 2]])
    trans[:, 2] = translation

    img_shear = cv.warpAffine(img, trans, (dst_w, dst_h))
    return img_shear


# this function implements a distortion of colour channels with a uniform random noise
def disturb_rgb_color_channels(img):
    img_corr = np.zeros(img.shape, dtype=np.uint8)
    for i in range(3):
        channel = img[:, :, i].astype(np.int32)  # extracting channel and converting it to signed integer values
        positive_mask = (channel > 0).astype(np.int32)
        new_channel = channel

        if np.random.random_sample() > 0.5:  # color correction
            new_channel = new_channel - 25 + 75 * np.random.rand() * positive_mask

        new_channel = np.maximum(new_channel, 0)  # clipping negative values
        new_channel = np.minimum(new_channel, 255)  # clipping too high values
        img_corr[:, :, i] = new_channel.astype(np.uint8)  # converting the result back to unsigned integer values
    return img_corr


# this function implements the distortion of colour channels along the principal intensity components
def disturb_pca_color_channels(img, eig_vals, eig_vecs):
    img_float = img.astype(np.single) / 255
    alpha = 0.35 * np.random.randn(3)
    alpha = np.minimum(alpha, 1)
    alpha = np.maximum(alpha, -1)
    eig_alpha = np.sqrt(eig_vals) * alpha
    color_correction = np.matmul(eig_vecs, eig_alpha)

    output_img_float = img_float
    for i in range(3):
        channel = output_img_float[:, :, i]
        channel_positive_mask = channel > 0
        channel[channel_positive_mask] = channel[channel_positive_mask] + color_correction[i]
        output_img_float[:, :, i] = channel

    output_img_float = np.maximum(output_img_float, 0)
    output_img_float = np.minimum(output_img_float, 1)
    output_img = 255 * output_img_float
    return output_img


# this function randomly flips the image
def flip_image(img):
    if np.random.random_sample() > 0.5:
        img_flip = cv.flip(img, 1)
    else:
        img_flip = img
    return img_flip


# this function implements the random stretching of the image
def stretch_image(img):
    a = math.log(1 / 1.27)
    b = math.log(1.27)
    scale1 = math.exp(a + (b - a) * np.random.random_sample())
    scale2 = math.exp(a + (b - a) * np.random.random_sample())
    new_width = round(scale1 * img.shape[1])
    new_height = round(scale2 * img.shape[0])
    img_stretch = cv.resize(img, (new_width, new_height))
    return img_stretch


# this function removes the layer of empty pixels around the rectangle containing the image data
def unpad_image(img):
    x, y, w, h = cv.boundingRect(cv.findNonZero(np.max(img, axis=2)))
    img_unpad = img[y:y + h, x:x + w, :].copy()
    return img_unpad

# this function brings the image to the same width and height
def square_image(img):
    im_size = img.shape[0:2]
    sqr_size = np.max(im_size)
    img_square = np.zeros((sqr_size, sqr_size, 3), dtype=np.uint8)
    if im_size[0] == im_size[1]:
        return img.copy()

    if im_size[0] < im_size[1]:
        ind1 = round((im_size[1] - im_size[0]) / 2) + 1
        ind2 = ind1 + im_size[0]
        ind3 = 0
        ind4 = im_size[1]
    else:
        ind1 = 0
        ind2 = im_size[0]
        ind3 = round((im_size[0] - im_size[1]) / 2) + 1
        ind4 = ind3 + im_size[1]
    img_square[ind1:ind2, ind3:ind4, :] = img
    return img_square


# this function applies the full data augmentation procedure to the image
def transform_image(img, eig_vals, eig_vecs, output_resolution):
    img_float = img.astype(np.single)
    disturb_choice = np.random.random_sample()
    if 0.3 < disturb_choice < 0.6:
        img_corr = disturb_rgb_color_channels(img_float)  # RGB color correction
    elif disturb_choice <= 0.3:
        img_corr = disturb_pca_color_channels(img_float, eig_vals, eig_vecs)
    else:
        img_corr = img_float.copy()

    img_float_gray = cv.cvtColor(img_float, cv.COLOR_BGR2GRAY)
    img_corr_gray = cv.cvtColor(img_corr, cv.COLOR_BGR2GRAY)

    eps = 3
    positive_mask = img_float_gray > 0
    positive_values = img_float_gray[positive_mask]
    corr_positive_values = img_corr_gray[positive_mask]

    corr_white_values = corr_positive_values[corr_positive_values > 255 - eps]
    corr_black_values = corr_positive_values[corr_positive_values < eps]

    white_fraction = len(corr_white_values) / len(positive_values)
    black_fraction = len(corr_black_values < eps) / len(positive_values)

    # as colour adjustment is a random process, we reject the transformation if the output
    # is too bright or too dark
    if white_fraction > 0.35:
        # print('WARNING: Color correction produced invalid results (white image)! The operation will be reverted!')
        img_corr = img_float.copy()

    if black_fraction > 0.35:
        # print('WARNING: Color correction produced invalid results (black image)! The operation will be reverted!')
        img_corr = img_float.copy()

    img_flip = flip_image(img_corr)  # random flipping
    rotate_angle = np.random.random_sample() * 360
    img_rotate = rotate_image(img_flip, rotate_angle)  # random rotation
    lambda1 = 0.17 * np.random.random_sample()
    lambda2 = 0.17 * np.random.random_sample()
    img_shear = shear_image(img_rotate, lambda1, lambda2)  # shearing
    img_stretch = stretch_image(img_shear)  # stretching
    img_unpad = unpad_image(img_stretch)  # unpadding
    img_square = square_image(img_unpad)  # squaring
    img_resize = cv.resize(img_square, output_resolution)  # resizing

    img_norm = img_resize.copy()
    img_norm = img_norm / 255.0
    return img_norm


# this function rescales the image to the given resolution
def format_image(img, output_resolution):
    img_float = img.astype(np.single)
    img_unpad = unpad_image(img_float)  # unpadding
    img_square = square_image(img_unpad)  # squaring
    img_resize = cv.resize(img_square, output_resolution)  # resizing
    return img_resize.astype(np.single) / 255.0


def generate_phantom(background_image, parasite_image, min_scale=0.8):
    image_flipped = flip_image(parasite_image)  # random flipping
    image_unpadded = unpad_image(image_flipped)

    width_ratio = background_image.shape[0] / image_unpadded.shape[0]
    if width_ratio < 1:
        image_unpadded = cv.resize(image_unpadded,
                                   (round(image_unpadded.shape[1] * width_ratio), background_image.shape[0]))
    height_ratio = background_image.shape[1] / image_unpadded.shape[1]
    if height_ratio < 1:
        image_unpadded = cv.resize(image_unpadded,
                                   (background_image.shape[1], round(image_unpadded.shape[0] * height_ratio)))

    scale_factor = min_scale + np.random.rand() * (1 - min_scale)
    image_unpadded = cv.resize(image_unpadded,
                               (round(scale_factor * image_unpadded.shape[1]),
                                round(scale_factor * image_unpadded.shape[0])
                                ))

    width_space = background_image.shape[0] - image_unpadded.shape[0]
    height_space = background_image.shape[1] - image_unpadded.shape[1]

    width_padding = round(np.random.rand() * width_space)
    height_padding = round(np.random.rand() * height_space)

    empty_mask = image_unpadded == 0

    phantom_image = (background_image[
                    width_padding:image_unpadded.shape[0]+width_padding,
                    height_padding:image_unpadded.shape[1]+height_padding,
                    :]).copy()
    phantom_image[empty_mask] = 0
    if phantom_image.max() == 0:
        print('Problem!')
    return phantom_image
