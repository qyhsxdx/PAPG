import numpy as np
import pandas as pd
import json
import os

from skimage.draw import line_aa, polygon, ellipse
from skimage.io import imread
import pylab as plt


MISSING_VALUE = -1

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result

def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
    yv, xv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                         sparse=False, indexing='xy')

    a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
    b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
    c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

    return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))

def make_limb_masks(joints, img_size):
    limbs = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]
    # limbs = [[0, 1, 14, 15, 16, 17], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    n_limbs = len(limbs)
    img_height, img_width = img_size[0], img_size[1]
    mask = np.zeros((img_height, img_width, n_limbs))

    # Gaussian sigma perpendicular to the limb axis.
    # sigma_perp = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 13]) ** 2
    sigma_perp = np.array([9,9,9,9,9,9,9,9,9,13]) ** 2


    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            missing = joints[limbs[i][j]][0] == MISSING_VALUE or joints[limbs[i][j]][1] == MISSING_VALUE
            if missing:
                break
            p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]
        if missing:
            continue
        if n_joints_for_limb == 4:
            p_top = np.mean(p[0:2, :], axis=0)
            p_bot = np.mean(p[2:4, :], axis=0)
            p = np.vstack((p_top, p_bot))

        center = np.mean(p, axis=0)

        sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.2])
        # sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 0.9])
        theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

        mask_i = make_gaussian_map(img_width, img_height, center, sigma_parallel, sigma_perp[i], theta)
        # mask_i = make_gaussian_map(img_width, img_height, center[::-1], sigma_perp[i], sigma_parallel, theta)
        mask[:, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

    return mask

def make_ms_limb_masks(joints, img_size, perpendicular=[5, 7, 9, 11, 13], parall=[1.5, 1.2, 0.9, 1.5, 1.2]):
    limbs = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]
    # limbs = [[0, 1, 14, 15, 16, 17], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    assert len(perpendicular) == len(parall)
    n_limbs = len(limbs)
    n_scale = len(perpendicular)
    img_height, img_width = img_size[0], img_size[1]
    mask = np.zeros((n_scale, img_height, img_width, n_limbs))

    # Gaussian sigma perpendicular to the limb axis.
    sigma_perp = np.tile(perpendicular, (n_limbs, 1))
    sigma_perp[-1, :] += 2
    sigma_perp = sigma_perp**2
    # sigma_perp = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 13]) ** 2
    # sigma_perp = np.array([7,7,7,7,7,7,7,7,7,9]) ** 2

    for scale_i in range(n_scale):
        for i in range(n_limbs):
            n_joints_for_limb = len(limbs[i])
            p = np.zeros((n_joints_for_limb, 2))

            for j in range(n_joints_for_limb):
                missing = joints[limbs[i][j]][0] == MISSING_VALUE or joints[limbs[i][j]][1] == MISSING_VALUE
                if missing:
                    break
                p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]
            if missing:
                continue
            if n_joints_for_limb == 4:
                p_top = np.mean(p[0:2, :], axis=0)
                p_bot = np.mean(p[2:4, :], axis=0)
                p = np.vstack((p_top, p_bot))

            center = np.mean(p, axis=0)

            sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / parall[scale_i]])            # sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 0.9])
            theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

            mask_i = make_gaussian_map(img_width, img_height, center, sigma_parallel, sigma_perp[:,scale_i][i], theta)
            # mask_i = make_gaussian_map(img_width, img_height, center[::-1], sigma_perp[i], sigma_parallel, theta)
            mask[scale_i, :, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

    return mask

def produce_ma_mask(kp_array, img_size, point_radius=1):
    from skimage.morphology import dilation, erosion, square
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
              [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
               [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    limbs = np.array(limbs) - 1
    for f, t in limbs:
        from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
        to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue

        norm_vec = kp_array[f] - kp_array[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


        vetexes = np.array([
            kp_array[f] + norm_vec,
            kp_array[f] - norm_vec,
            kp_array[t] - norm_vec,
            kp_array[t] + norm_vec
        ])
        yy, xx = polygon(vetexes[:, 0], vetexes[:, 1], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(kp_array):
        if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            continue
        yy, xx = ellipse(joint[0], joint[1], point_radius, point_radius, shape=img_size)
        mask[yy, xx] = True

    # mask = dilation(mask, square(5))
    # mask = erosion(mask, square(5))
    mask = np.expand_dims(mask, -1).astype(np.uint8)
    return mask

def compute_pose(image_dir, annotations_file, savePath):
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')
    image_size = (288, 144)
    cnt = len(annotations_file)
    for i in range(cnt):
        print('processing %d / %d ...' %(i, cnt))
        row = annotations_file.iloc[i]
        name = row.name
        print(savePath, name)
        file_name = os.path.join(savePath, str(name) + "_" + str(i) + '.npy')
        kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        pose = cords_to_map(kp_array, image_size)
        if pose.sum() == 0:
            continue
        else:
            np.save(file_name, pose)


def save_pose_coor(annotations_file, save_name):
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')
    cnt = len(annotations_file)
    kp_list = {}
    for i in range(cnt):
        print('processing %d / %d ...' % (i, cnt))
        row = annotations_file.iloc[i]
        name = row.name
        kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        kp_list[name] = kp_array
    print('save finished!')
    np.save(save_name, kp_list)


if __name__ == '__main__':
    img_dir = './train_ir/'  # raw image path
    #img_dir = "./train_rgb/"
    annotations_file = './sysu-mm01-annotation-train_ir.csv'  # pose annotation path
    #annotations_file = './sysu-mm01-annotation-train_rgb.csv'
    save_path = './train_ir_pose_map/'  # path to store pose maps
    #save_path = './train_rgb_pose_map/'  # path to store pose maps
    compute_pose(img_dir, annotations_file, save_path)

