import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
import cv2
import os
import argparse


def make_grayscale_diff_data(path, num_channels, y_data):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    data_path = path[:-4] + '/'
    row, col = 192, 256

    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    if y_data == 'angle':
        y = np.zeros((num_rows - num_channels, 1), dtype=np.float64)
    else:
        y = np.zeros((num_rows - num_channels, 2), dtype=np.float64)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print("Processed {} images out of {} images".format(i, str(num_rows)))
        for j in range(num_channels):
            path0 = df['filename'].iloc[i - j - 1]
            path1 = df['filename'].iloc[i - j]
            img0 = cv2.imread(data_path + path0, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(data_path + path1, cv2.IMREAD_GRAYSCALE)
            img0 = cv2.resize(img0, (256, 192))
            img1 = cv2.resize(img1, (256, 192))
            img0 = img0.astype('float32')
            img1 = img1.astype('float32')
            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)
            X[i - num_channels, :, :, j] = img
        if y_data == 'angle':
            y[i - num_channels, :] = [df["angle"].iloc[i]]
        else:
            y[i - num_channels, :] = [df["angle"].iloc[i], df["speed"].iloc[i]]

    return X, y


def load_data(data_dir):
    X_train1 = np.load(os.path.join(data_dir, "X_train_part1.npy"))
    y_train1 = np.load(os.path.join(data_dir, "y_train_part1.npy"))
    X_train2 = np.load(os.path.join(data_dir, "X_train_part2.npy"))
    y_train2 = np.load(os.path.join(data_dir, "y_train_part2.npy"))
    X_train4 = np.load(os.path.join(data_dir, "X_train_part4.npy"))
    y_train4 = np.load(os.path.join(data_dir, "y_train_part4.npy"))
    X_train5 = np.load(os.path.join(data_dir, "X_train_part5.npy"))
    y_train5 = np.load(os.path.join(data_dir, "y_train_part5.npy"))
    X_train6 = np.load(os.path.join(data_dir, "X_train_part6.npy"))
    y_train6 = np.load(os.path.join(data_dir, "y_train_part6.npy"))

    X_train = np.concatenate((X_train1, X_train2, X_train4, X_train5, X_train6), axis=0)
    y_train = np.concatenate((y_train1, y_train2, y_train4, y_train5, y_train6), axis=0)
    X_test = np.concatenate((X_train1, X_train2, X_train4, X_train5, X_train6), axis=0)
    y_test = np.concatenate((y_train1, y_train2, y_train4, y_train5, y_train6), axis=0)

    y_train = np.reshape(y_train, [-1, 1])
    y_test = np.reshape(y_test, [-1, 1])

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--y", type=str, help="angle|speed",
                    choices=["angle", "speed"], required=True)
    args = ap.parse_args()
    args = vars(args)

    print("Pre-processing udacity data...")
    for i in [1, 2, 4, 5, 6]:
        X_train, y_train = make_grayscale_diff_data('udacity/' + str(i) + 's.csv', 4, args['y'])
        np.save("./data/X_train_part{}".format(i), X_train)
        np.save("./data/y_train_part{}".format(i), y_train)
