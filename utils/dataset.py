import csv

import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

from transformer.Constants import VOCABULARY, BOS, EOS, PAD


class LipReadingDataSet(Dataset):
    def __init__(self, index_file, transforms=None):
        self.index = []
        with open(index_file) as f:
            reader = csv.reader(f)
            for mp4_name, target in reader:
                self.index.append((mp4_name, target))
        self.transforms = transforms

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        mp4_name, sentence = self.index[item]

        images = LipReadingDataSet._load_mp4(mp4_name)
        target = LipReadingDataSet._get_target(sentence)
        if self.transforms:
            images = self.transforms(images)
        return images, target, sentence

    @staticmethod
    def _load_mp4(mp4_name):
        video_cap = cv2.VideoCapture(mp4_name)
        frames = []
        while True:
            success, frame = video_cap.read()
            if not success:
                break
            frame = np.transpose(frame, [2, 0, 1])
            frames.append(frame)
        return frames

    @staticmethod
    def _get_target(sentence):
        return [BOS] + [VOCABULARY.index(char) for char in sentence] + [EOS]


def _get_position_batch(seq_lengths, max_length):
    position_batch = [list(range(1, length + 1)) for length in seq_lengths]
    for position in position_batch:
        position.extend([PAD] * (max_length - len(position)))
    return np.asarray(position_batch)


def _get_images_list_batch(images_list_batch):
    seq_lengths = [len(images_list) for images_list in images_list_batch]
    image_shape, dtype = images_list_batch[0][0].shape, images_list_batch[0][0].dtype
    max_length = max(seq_lengths)
    for images_list in images_list_batch:
        images_list.extend([np.zeros(image_shape, dtype=dtype)] * (max_length - len(images_list)))
    images_position_batch = _get_position_batch(seq_lengths, max_length)

    images_list_batch = np.asarray(images_list_batch)
    images_position_batch = np.asarray(images_position_batch)
    return images_list_batch, images_position_batch


def _get_target_batch(target_batch):
    seq_lengths = [len(target) for target in target_batch]
    max_length = max(seq_lengths)
    for target in target_batch:
        target.extend([PAD] * (max_length - len(target)))
    target_position_batch = _get_position_batch(seq_lengths, max_length)

    target_batch = np.asarray(target_batch, dtype=np.int64)
    target_position_batch = np.asarray(target_position_batch)
    return target_batch, target_position_batch


def collate_fn(batch):
    images_list_batch, target_batch, sentence_batch = zip(*batch)
    images_list_batch = list(images_list_batch)
    target_batch = list(target_batch)
    sentence_batch = list(sentence_batch)
    images_list_batch, images_position_batch = _get_images_list_batch(images_list_batch)
    target_batch, target_position_batch = _get_target_batch(target_batch)
    images_list_batch = torch.from_numpy(images_list_batch).float().div(255)
    target_batch = torch.from_numpy(target_batch)
    images_position_batch = torch.from_numpy(images_position_batch)
    target_position_batch = torch.from_numpy(target_position_batch)
    return (images_list_batch, images_position_batch), (target_batch, target_position_batch), sentence_batch
