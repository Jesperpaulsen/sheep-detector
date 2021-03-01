# Script to create the test/train dataset
from config import retinanet_config as config
import argparse
import random
import pandas as pd
import os
import csv
from torchvision import datasets, transforms as T
import torch
from SheepDataset import SheepDataset


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-a", "--annotations", default=config.OPTICAL_CSV,
                             help='path to annotations')
argument_parser.add_argument("-i", "--images", default=config.IMAGES_PATH,
                             help="path to images")
argument_parser.add_argument("-t", "--train", default=config.TRAIN_CSV,
                             help="path to output training CSV file")
argument_parser.add_argument("-e", "--test", default=config.TEST_CSV,
                             help="path to output test CSV file")
argument_parser.add_argument("-c", "--classes", default=config.CLASSES_CSV,
                             help="path to output classes CSV file")
argument_parser.add_argument("-s", "--split", type=float, default=config.TRAIN_TEST_SPLIT,
                             help="train and test split")
args = vars(argument_parser.parse_args())

annot_path = args['annotations']
images_path = args['images']
train_csv = args['train']
test_csv = args['test']
classes_csv = args['classes']
train_test_split = args['split']

dataset = SheepDataset(annot_path=annot_path, images_path=images_path)
print(dataset.__getitem__(1))

