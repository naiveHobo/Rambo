import os
import argparse
from dataset import load_data
from configuration import Configuration
from model import Rambo
import numpy as np
import pandas as pd
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("--mode", type=str, help="train|test",
                choices=["train", "test"], required=True)
ap.add_argument("--model", type=str, help="rambo|nvidia1|nvidia2|comma",
                choices=["rambo", "nvidia1", "nvidia2", "comma"], default="rambo")
ap.add_argument("--resume", action="store_true",
                help="resume training from last checkpoint")
ap.add_argument("--train_dir", default="./train",
                help="path to the directory where checkpoints are to be stored")
ap.add_argument("--data_dir", default="./data",
                help="path to the directory where the data is stored")
ap.add_argument("--log_dir", default="./graph",
                help="path to the directory where tensorboard logs are to be written")
ap.add_argument("--epochs", type=int, default=25,
                help="number of epochs")
ap.add_argument("--batch_size", type=int, default=32,
                help="size of mini-batch")
ap.add_argument("--learning_rate", type=float, default=0.001,
                help="learning rate")
ap.add_argument("--dropout", action="store_true",
                help="use dropout")
ap.add_argument("--batch_norm", action="store_true",
                help="use batch normalization")
ap.add_argument("--beta_l2", type=float, default=0.0,
                help="define beta for l2 regularization")
ap.add_argument("--visualize", action="store_true",
                help="add visualization graph")
ap.add_argument("--save_model", action="store_true",
                help="save model for inference")

args = ap.parse_args()
config = Configuration(vars(args))

if config.mode == "train":
    data = load_data(config.data_dir)
    model = Rambo(config, data=data)
    model.train()

if config.mode == "test":
    model = Rambo(config)
    data = load_data(config.data_dir)

    df = pd.read_csv('/home/naivehobo/Desktop/udacity/1.csv')
    smoothed_angle = 0.
    for i in range(len(data[0])-1):
        steering, mask = model.predict(data[0][i:i+1], True)
        green = np.zeros((192,256,3), np.uint8)
        green[:,:] = (0,255,0)
        green = mask * green
        path0 = df['filename'].iloc[i+4]
        img = cv2.imread('/home/naivehobo/Desktop/udacity/1/'+path0)
        mask = img*mask
        # cv2.addWeighted(green, 0.8, img, 0.2, 0, img)
        cv2.imshow('yolo', img)
        cv2.imshow('yolo1', mask)

        wheel = cv2.imread('steering_wheel_image.png')
        rows, cols, _ = wheel.shape
        angle = steering * 180 / np.pi
        if angle - smoothed_angle == 0:
            continue
        smoothed_angle += 0.2 * pow(abs((angle - smoothed_angle)), 2.0 / 3.0) * (
                angle - smoothed_angle) / abs(angle - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), smoothed_angle, 1)
        dst = cv2.warpAffine(wheel, M, (cols, rows))
        cv2.imshow("Pedicted", dst)
        if cv2.waitKey(1) == ord('q'):
            quit()
