from torch.utils.data import Dataset
import os
import PIL
from PIL import Image
import numpy as np
from torchvision import transforms

class CelebAHQ(Dataset):
    def __init__(self, data_root, interpolation="bicubic", flip_p=0.5, size=256):
        #self.data_root = data_root
        #self.img_names = os.listdir(self.data_root)
        #self.img_paths = [os.path.join(self.data_root, names) for names in self.img_names]
        self.img_names = data_root
        self.img_paths = data_root
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        image_path = self.img_paths
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        output = image
        # output = {}
        # output['image'] = image
        # output['image_path'] = image_path
        return output