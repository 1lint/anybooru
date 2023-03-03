from pathlib import Path
from PIL import Image
import torchvision
import random

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial
from multiprocessing import cpu_count


class PNGDataset(Dataset):
    def __init__(
        self,
        data_dir,
        ucg=0.10,
        prompt_key="tags",
        resolution=(512, 512),
        cond_key="cond",
        target_key="image",
    ):
        super().__init__()
        vars(self).update(locals()) # attach params to self

        self.img_paths = list(Path(data_dir).glob("*.png"))

        # ucg and transforms from waifu diffusion
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
                torchvision.transforms.Resize(resolution),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)

        if self.prompt_key not in image.info:
            print(f"{img_path} does lacks {self.prompt_key}, skipping to next image")
            return self.__getitem__(idx + 1 % len(self))
        elif random.random() > self.ucg:
            tags = image.info[self.prompt_key]
        else:
            tags = ""

        target = self.transforms(image)

        return {self.target_key: target, self.cond_key: tags}


class PNGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=1,
        train_dir=None,
        val_dir=None,
        test_dir=None,
        predict_dir=None,
        num_workers=None,
        persistent_workers=True,
        **kwargs,  # passed to dataset class
    ):
        super().__init__()
        vars(self).update(locals()) # attach params to self

        if num_workers is None:
            num_workers = max(cpu_count() // 2, 1)

        self.ds_wrapper = partial(PNGDataset, **kwargs)

        self.dl_wrapper = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.ds_wrapper(data_dir=self.train_dir)
            self.val_dataset = self.ds_wrapper(data_dir=self.val_dir)

        elif stage == "predict":
            self.predict_dataset = self.ds_wrapper(data_dir=self.predict_dir)

        elif stage == "test":
            self.test_dataset = self.ds_wrapper(data_dir=self.test_dir)

        else:
            raise Exception(f"Error: unhandled stage {stage}")

    def train_dataloader(self):
        return self.dl_wrapper(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.dl_wrapper(self.val_dataset, shuffle=False)

    def predict_dataloader(self):
        return self.dl_wrapper(self.predict_dataset, shuffle=False)

    def test_dataloader(self):
        return self.dl_wrapper(self.test_dataset, shuffle=False)
