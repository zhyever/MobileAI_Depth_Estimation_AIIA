import numpy as np
from utils import DepthNorm
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
from augment import BasicPolicy, ImageNetPolicy
import csv
import tensorflow as tf
from tensorflow.keras import layers

def resize(img, resolution=480):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution * 4 / 3)), preserve_range=True, mode='reflect', anti_aliasing=True)

# width 640
# height 480
def get_data(batch_size):
    # data = extract_zip(nyu_data_zipfile)

    with open("./train_filename.csv", "r") as f:
        reader = csv.reader(f)
        nyu2_train = list([[row[0], row[1]] for row in reader])

    with open("../depth_pytorch/val_filename.csv", "r") as f:
        reader = csv.reader(f)
        nyu2_test = list([[row[0], row[1]] for row in reader])

    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 480, 640, 1)

    # Helpful for testing...
    # if False:
    #     nyu2_train = nyu2_train[:10]
    #     nyu2_test = nyu2_test[:10]

    return '', nyu2_train, nyu2_test, shape_rgb, shape_depth


def get_train_test_data(batch_size):
    data, train, test, shape_rgb, shape_depth = get_data(batch_size)

    train_generator = BasicAugmentRGBSequence(data, train, batch_size=batch_size, shape_rgb=shape_rgb,
                                              shape_depth=shape_depth, is_flip=True,
                                              is_addnoise=True, is_erase=True)
    test_generator = BasicRGBSequence(data, test, batch_size=batch_size, shape_rgb=shape_rgb,
                                          shape_depth=shape_depth)

    return train_generator, test_generator


def deal_y(y, maxdepth):
    return maxdepth / y

class BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False,
                 is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy(color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
                                  add_noise_peak=0 if not is_addnoise else 20,
                                  erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros(self.shape_rgb), np.zeros(self.shape_depth)

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N - 1)
            # print('index=', index)

            x = np.clip(np.asarray(Image.open(self.dataset[index][0])) / 255, 0, 1)
            y = np.asarray(Image.open(self.dataset[index][1]))
            y = np.array(y)
            y[y<1] = 1 # [1~45000]

            y = np.reshape(y, (480, 640, 1))

            # print(y)

            batch_x[i] = x
            batch_y[i] = y

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

        return batch_x, batch_y


class BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros(self.shape_rgb), np.zeros(self.shape_depth)
        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N - 1)

            x = np.clip(np.asarray(Image.open(self.dataset[index][0])) / 255, 0, 1)
            y = np.asarray(Image.open(self.dataset[index][1]))
            y = np.array(y)
            y[y < 1] = 45000

            y = np.reshape(y, (480, 640, 1))

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, batch_y
