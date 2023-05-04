import numpy as np
import h5py
import os

# Modify next 2 lines
data_path = "data/nyu"  # where to find nyu_depth_v2_labeled.mat and splits.mat
output_path = "data/nyu"  # where to put the resulting dataset

train_path = os.path.join(output_path, "train/official/")
val_path = os.path.join(output_path, "val/official/")
test_path = os.path.join(output_path, "test/official/")

f = h5py.File(data_path + "/nyu_depth_v2_labeled.mat")
images = f["images"]
depths = f["depths"]
labels = f["labels"]

images = np.array(images)
depths = np.array(depths)
labels = np.array(labels)

# set random seed
seed = 1
np.random.seed(seed)

# create 80-10-10 train-val-test split
idx = np.arange(1, len(images) + 1)
shuffle_idx = np.random.permutation(idx)
train_idx = shuffle_idx[:int(0.8 * len(images))]
val_idx = shuffle_idx[int(0.8 * len(images)):int(0.9 * len(images))]
test_idx = shuffle_idx[int(0.9 * len(images)):]

if not os.path.isdir(train_path):
    os.makedirs(train_path)

if not os.path.isdir(val_path):
    os.makedirs(val_path)

if not os.path.isdir(test_path):
    os.makedirs(test_path)

for idx in range(len(train_idx)):
    f_idx = "{0:0>5}".format(int(train_idx[idx]))
    print("train:", f_idx)
    h5f = h5py.File(train_path + f_idx + ".h5", "w")

    h5f["rgb"] = np.transpose(images[train_idx[idx] - 1][0], (0, 2, 1))
    h5f["depth"] = np.transpose(depths[train_idx[idx] - 1][0], (1, 0))

    h5f.close()

for idx in range(len(val_idx)):
    f_idx = "{0:0>5}".format(int(val_idx[idx]))
    print("val:", f_idx)
    h5f = h5py.File(val_path + f_idx + ".h5", "w")

    h5f["rgb"] = np.transpose(images[val_idx[idx] - 1][0], (0, 2, 1))
    h5f["depth"] = np.transpose(depths[val_idx[idx] - 1][0], (1, 0))

    h5f.close()

for idx in range(len(test_idx)):
    f_idx = "{0:0>5}".format(int(test_idx[idx]))
    print("test:", f_idx)
    h5f = h5py.File(test_path + f_idx + ".h5", "w")

    h5f["rgb"] = np.transpose(images[test_idx[idx] - 1][0], (0, 2, 1))
    h5f["depth"] = np.transpose(depths[test_idx[idx] - 1][0], (1, 0))

    h5f.close()

print(train_idx[0])
print(images[train_idx[0] - 1][0].shape)
print(depths[train_idx[0] - 1][0].shape)
