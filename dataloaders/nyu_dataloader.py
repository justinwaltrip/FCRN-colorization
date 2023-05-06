import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader
import torchvision as tv
from PIL import Image

iheight, iwidth = 480, 640  # raw image size

to_tensor = transforms.ToTensor()
from_tensor = transforms.FromTensor()


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

    def __init__(
        self, root, type, sparsifier=None, modality="rgb", small_subset: bool = False
    ):
        super(NYUDatasetColorization, self).__init__(
            root, type, sparsifier, modality, small_subset=small_subset
        )
        self.output_size = (228, 304)

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            grayscale_np, rgb_np = self.transform(rgb, depth)
        else:
            raise (RuntimeError("transform not defined"))

        if self.modality == "rgb":
            input_np = grayscale_np
        elif self.modality == "rgbd":
            input_np = self.create_rgbd(grayscale_np, rgb_np)
        elif self.modality == "d":
            input_np = self.create_sparse_depth(grayscale_np, rgb_np)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        target_tensor = to_tensor(rgb_np)
        # depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, target_tensor

    def train_transform(self, rgb, depth):
        # if using small subset, skip data augmentation
        if self.small_subset:
            return self.val_transform(rgb, depth)

        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose(
            [
                transforms.Resize(
                    250.0 / iheight
                ),  # this is for computational efficiency, since rotation can be slow
                transforms.Rotate(angle),
                transforms.CenterCrop(self.output_size),
                transforms.HorizontalFlip(do_flip),
            ]
        )
        grayscale_transform = tv.transforms.Grayscale(num_output_channels=3)

        # perform resize, rotation, crop, flip, and jitter on rgb
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering

        # further transform rgb to gray scale
        grayscale_np = grayscale_transform(Image.fromarray(rgb_np))

        # normalize
        rgb_np = np.asfarray(rgb_np, dtype="float") / 255
        grayscale_np = np.asfarray(grayscale_np, dtype="float") / 255

        # input gray scale, output rgb
        return grayscale_np, rgb_np

    def val_transform(self, rgb, depth):
        transform = transforms.Compose(
            [
                transforms.Resize(240.0 / iheight),
                transforms.CenterCrop(self.output_size),
            ]
        )
        grayscale_transform = tv.transforms.Grayscale(num_output_channels=3)

        # perform resize, crop on rgb
        rgb_np = transform(rgb)

        # further transform rgb to gray scale
        grayscale_np = grayscale_transform(Image.fromarray(rgb_np))

        # normalize
        rgb_np = np.asfarray(rgb_np, dtype="float") / 255
        grayscale_np = np.asfarray(grayscale_np, dtype="float") / 255

        # input gray scale, output rgb
        return grayscale_np, rgb_np

    # def inv_val_transform(self, rgb_np, depth_np):
    #     rgb_np = rgb_np * 255
    #     inv_transform = transforms.Compose(
    #         [
    #             transforms.Resize(iheight / 240.0),
    #         ]
    #     )
    #     rgb_np = np.asarray(rgb_np, dtype="uint8")
    #     rgb = inv_transform(rgb_np)
    #     depth = inv_transform(depth_np[:, :, 0])
    #     return rgb, depth
