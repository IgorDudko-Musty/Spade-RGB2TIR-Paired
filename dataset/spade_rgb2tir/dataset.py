from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class RGB2TIRData(Dataset):
    def __init__(self, path):
        super().__init__()
        train_RGB = Path(path) / "train_RGB"
        train_TIR = Path(path) / "train_TIR"

        self.train_RGB = [img for img in train_RGB.iterdir()]
        self.train_TIR = [img for img in train_TIR.iterdir()]

    def __getitem__(self, index):
        rgb_path = self.train_RGB[index]
        tir_path = self.train_TIR[index]

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tir = cv2.imread(tir_path, cv2.IMREAD_GRAYSCALE)

        return rgb, tir

    def __len__(self):
        return len(self.train_RGB)


class FLIRAlignedData(Dataset):
    def __init__(
        self,
        path=r"/home/igordudko/datasets/RGB2TIR_PAIRED/Aligned_FLIR_dataset_git/align/JPEGImages",
    ):
        super().__init__()
        path = Path(path)
        self.train_RGB = []
        self.train_TIR = []
        for file_name in path.iterdir():
            if "RGB" in file_name.name:
                id_num = file_name.name.split("_")[1]
                self.train_RGB.append(file_name)
                self.train_TIR.append(path / f"FLIR_{id_num}_PreviewData.jpeg")

    def __getitem__(self, index):
        rgb_path = self.train_RGB[index]
        tir_path = self.train_TIR[index]

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1).to(torch.float32) / 127.5 - 1.0

        tir = cv2.imread(tir_path, cv2.IMREAD_GRAYSCALE)
        tir = torch.from_numpy(tir)
        tir = tir.unsqueeze(0).to(torch.float32) / 127.5 - 1.0

        return rgb, tir

    def __len__(self):
        return len(self.train_RGB)


class LLVIPData(Dataset):
    def __init__(
        self,
        path=r"/home/igordudko/datasets/RGB2TIR_PAIRED",
    ):
        super().__init__()
        path_vis = Path(path) / "visible"
        path_tir = Path(path) / "infrared"
        self.train_RGB = []
        self.train_TIR = []
        for split in ["train", "test"]:
            vis_dir = path_vis / split
            tir_dir = path_tir / split
            for file_name in vis_dir.iterdir():
                self.train_RGB.append(file_name)
                self.train_TIR.append(tir_dir / file_name.name)

    def __getitem__(self, index):
        rgb_path = self.train_RGB[index]
        tir_path = self.train_TIR[index]

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tir = cv2.imread(tir_path, cv2.IMREAD_GRAYSCALE)

        return rgb, tir

    def __len__(self):
        return len(self.train_RGB)
