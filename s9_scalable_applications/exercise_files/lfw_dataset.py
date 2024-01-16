"""LFW dataloading."""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = 'tight'

ImageFile.LOAD_TRUNCATED_IMAGES = True

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class LFWDataset(Dataset):
    """Initialize LFW dataset."""
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you needs
        self.imgs = []
        self.labels = []
        c = 0
        for folder in os.listdir(path_to_folder):
            for img in os.listdir(os.path.join(path_to_folder, folder)):
                img_data = Image.open(os.path.join(path_to_folder, folder, img))
                # convert img_data to torch tensor:
                # img_data = np.array(img_data)
                # img_data = torch.from_numpy(img_data)

                self.imgs.append(img_data)
                self.labels.append(folder)

            if c > 16000:
                break
            else:
                c += 1

        # print(plt.imshow(self.imgs[0]))

        self.transform = transform

    def __len__(self):
        """Return length of dataset."""
        return len(self.imgs)
        # return None  # TODO: fill out

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get item from dataset."""
        # TODO: fill out
        return self.transform(self.imgs[index]), self.labels[index]
        # return self.transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="", type=str)
    parser.add_argument("-batch_size", default=512, type=int)
    parser.add_argument("-num_workers", default=None, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose([transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()])

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.visualize_batch:
        # Visualize first batch
        for batch_idx, _batch in enumerate(dataloader):
            # if batch_idx > 0:
            #     break
            # print(_batch[0])
            # print(_batch[0].dtype) # torch.float32
            # print(_batch[0].shape) # torch.Size([8, 1, 250, 250])
            # print(type(_batch[0])) # <class 'torch.Tensor'>
            _batch[0] = _batch[0] * 255
            _batch[0] = _batch[0].to(torch.uint8)
            list_of_tensors = [tensor for tensor in _batch[0]]
            grid = make_grid(list_of_tensors, nrow=10)
            show(grid)
            plt.show()

    if args.get_timing:
        # lets do some repetitions
        res = []
        for _ in range(5):
            start = time.time()
            for batch_idx, _batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print(f"Timing: {np.mean(res)}+-{np.std(res)}")
