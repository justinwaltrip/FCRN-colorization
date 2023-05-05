import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640  # raw image size


class NYUDataset(MyDataloader):
    def __init__(
        self, root, type, sparsifier=None, modality="rgb", small_subset: bool = False
    ):
        super(NYUDataset, self).__init__(
            root, type, sparsifier, modality, small_subset=small_subset
        )
        self.output_size = (228, 304)

    def train_transform(self, rgb, depth):
        # if using small subset, skip data augmentation
        if self.small_subset:
            return self.val_transform(rgb, depth)

        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose(
            [
                transforms.Resize(
                    250.0 / iheight
                ),  # this is for computational efficiency, since rotation can be slow
                transforms.Rotate(angle),
                transforms.Resize(s),
                transforms.CenterCrop(self.output_size),
                transforms.HorizontalFlip(do_flip),
            ]
        )
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype="float") / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose(
            [
                transforms.Resize(240.0 / iheight),
                transforms.CenterCrop(self.output_size),
            ]
        )
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype="float") / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def inv_val_transform(self, rgb_np, depth_np):
        rgb_np = rgb_np * 255
        inv_transform = transforms.Compose(
            [
                transforms.Resize(iheight / 240.0),
            ]
        )
        rgb_np = np.asarray(rgb_np, dtype="uint8")
        rgb = inv_transform(rgb_np)
        depth = inv_transform(depth_np[:, :, 0])
        return rgb, depth



class NYUDatasetColorization(MyDataloader):
    # TODO update for colorization task

    def __init__(
        self, root, type, sparsifier=None, modality="rgb", small_subset: bool = False
    ):
        super(NYUDataset, self).__init__(
            root, type, sparsifier, modality, small_subset=small_subset
        )
        self.output_size = (228, 304)

    def train_transform(self, rgb, depth):
        # if using small subset, skip data augmentation
        if self.small_subset:
            return self.val_transform(rgb, depth)

        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose(
            [
                transforms.Resize(
                    250.0 / iheight
                ),  # this is for computational efficiency, since rotation can be slow
                transforms.Rotate(angle),
                transforms.Resize(s),
                transforms.CenterCrop(self.output_size),
                transforms.HorizontalFlip(do_flip),
            ]
        )
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype="float") / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose(
            [
                transforms.Resize(240.0 / iheight),
                transforms.CenterCrop(self.output_size),
            ]
        )
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype="float") / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def inv_val_transform(self, rgb_np, depth_np):
        rgb_np = rgb_np * 255
        inv_transform = transforms.Compose(
            [
                transforms.Resize(iheight / 240.0),
            ]
        )
        rgb_np = np.asarray(rgb_np, dtype="uint8")
        rgb = inv_transform(rgb_np)
        depth = inv_transform(depth_np[:, :, 0])
        return rgb, depth
