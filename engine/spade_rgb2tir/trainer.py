import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import wandb
from dataset.spade_rgb2tir.dataset import FLIRAlignedData
from engine.spade_rgb2tir.optim.optimizers import create_optimizer
from engine.spade_rgb2tir.optim.schedulers import create_scheduler
from models.spade_rgb2tir.registry import create_model


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        wandb_cfg = self.cfg.get("wandb", {})
        self.wandb_log = wandb.init(
            **wandb_cfg["wandb"],
            config={
                "generator": self.cfg["model"]["generator"],
                "discriminator": self.cfg["model"]["discriminator"],
                "losses": self.cfg["model"]["losses"],
                "optim_D": self.cfg["optim"]["optimizer"]["discriminator"],
                "optim_G": self.cfg["optim"]["optimizer"]["generator"],
                "scheduler_D": self.cfg["optim"]["scheduler"]["discriminator"],
                "scheduler_G": self.cfg["optim"]["scheduler"]["generator"],
                "train": self.cfg["train"]["train"],
                "data": self.cfg["data"]["data"],
            },
        )

        self.device = torch.device(self.cfg["train"]["train"]["device"])

        self.generator, self.discriminator, self.losses = create_model(cfg)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        for k, loss_fn in self.losses.items():
            self.losses[k] = loss_fn.to(self.device)

        self.optim_D = create_optimizer(
            self.discriminator.parameters(),
            self.cfg["optim"]["optimizer"]["discriminator"],
        )
        self.optim_G = create_optimizer(
            self.generator.parameters(), self.cfg["optim"]["optimizer"]["generator"]
        )

        self.sched_D = create_scheduler(
            self.optim_D, self.cfg["optim"]["scheduler"]["discriminator"]
        )
        self.sched_G = create_scheduler(
            self.optim_G, self.cfg["optim"]["scheduler"]["generator"]
        )

        self.dataloader = DataLoader(
            FLIRAlignedData(self.cfg["data"]["data"]["root"]),
            batch_size=self.cfg["data"]["data"]["batch_size"],
            shuffle=self.cfg["data"]["data"]["shuffle"],
            num_workers=self.cfg["data"]["data"]["num_workers"],
        )

    def train_one_step(self, batch):
        img_rgb, img_tir = batch
        img_rgb = img_rgb.to(self.device)
        img_tir = img_tir.to(self.device)
        #### DISCRIMINATOR LEARNING #############
        self.optim_D.zero_grad()
        fake = self.generator(img_rgb)

        prob_map_fake, _ = self.discriminator(fake.detach())
        prob_map_tir, _ = self.discriminator(img_tir)

        loss_D = 0.0
        for pmf, pmt in zip(prob_map_fake, prob_map_tir):
            loss_D = loss_D + self.losses["gan"].D_loss(pmt, pmf)
        loss_D = loss_D / len(prob_map_tir)

        loss_D.backward()
        self.optim_D.step()

        #### GENERATOR LEARNING #############
        self.optim_G.zero_grad()
        fake = self.generator(img_rgb)

        prob_map_fake, feats_fake = self.discriminator(fake)
        _, feats_tir = self.discriminator(img_tir.detach())

        loss_G = 0.0
        for prob_fake in prob_map_fake:
            loss_G = loss_G + self.losses["gan"].G_loss(prob_fake)
        loss_G = loss_G / len(prob_map_fake)

        loss_l1 = self.losses["l1"](fake, img_tir)

        loss_FM = 0
        for ff, ft in zip(feats_fake, feats_tir):
            loss_FM += self.losses["fm"](ft, ff)
        loss_FM = loss_FM / len(feats_fake)

        loss_G_tot = loss_G + loss_l1 + loss_FM

        loss_G_tot.backward()
        self.optim_G.step()
        return (
            loss_G_tot.item(),
            loss_G.item(),
            loss_D.item(),
            loss_l1.item(),
            loss_FM.item(),
            fake.mean().item(),
            fake.std().item(),
            img_tir.mean().item(),
            img_tir.std().item(),
            fake.detach(),
        )

    def train_one_epoch(self, dataloader, epoch, total_epochs):
        pbar = tqdm.tqdm(
            dataloader,
            dynamic_ncols=True,
            desc=f"Epoch {epoch}/{total_epochs}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        count = 0
        for batch in pbar:
            count += 1
            (
                loss_G_tot,
                loss_G,
                loss_D,
                loss_l1,
                loss_FM,
                f_m,
                f_std,
                tir_m,
                tir_std,
                fake,
            ) = self.train_one_step(batch=batch)

            pbar.set_postfix(
                {
                    "tot_G": f"{loss_G_tot:.3f}",
                    "G": f"{loss_G:.3f}",
                    "D": f"{loss_D:.5f}",
                    "L1": f"{loss_l1:.3f}",
                    "FM": f"{loss_FM:.3f}",
                    "f_m": f"{f_m:.3f}",
                    "f_std": f"{f_std:.3f}",
                    "tir_m": f"{tir_m:.3f}",
                    "tir_std": f"{tir_std:.3f}",
                    "lr_D": f"{self.optim_D.param_groups[0]['lr']:.8f}",
                    "lr_G": f"{self.optim_G.param_groups[0]['lr']:.8f}",
                }
            )
            if count % 200 == 0:
                self.wandb_log.log(
                    {
                        "loss/tot_G": loss_G_tot,
                        "loss/G": loss_G,
                        "loss/D": loss_D,
                        "loss/L1": loss_l1,
                        "loss/FM": loss_FM,
                        "stats/fake_mean": f_m,
                        "stats/fake_std": f_std,
                        "stats/tir_mean": tir_m,
                        "stats/tir_std": tir_std,
                    }
                )

        log = (
            f"Epoch={epoch}, "
            f"tot_G={loss_G_tot}, "
            f"G={loss_G}, "
            f"D={loss_D}, "
            f"L1={loss_l1}, "
            f"FM={loss_FM}, "
            f"lr_D={self.optim_D.param_groups[0]['lr']}, "
            f"lr_G={self.optim_G.param_groups[0]['lr']}, "
            f"fake={[f_m, f_std]}, "
            f"tir_true={[tir_m, tir_std]}\n"
        )
        self.sched_D.step()
        self.sched_G.step()
        self.save_log(log)

        self.wandb_log.log(
            {
                "lr/G": self.optim_G.param_groups[0]["lr"],
                "lr/D": self.optim_D.param_groups[0]["lr"],
            }
        )

        return fake, batch[1]

    def fit(self):
        for epoch in range(1, self.cfg["train"]["train"]["epochs"] + 1):
            fake, tir = self.train_one_epoch(
                self.dataloader, epoch, self.cfg["train"]["train"]["epochs"]
            )
            self.save_checkpoit(
                self.generator,
                self.discriminator,
                self.cfg["train"]["train"]["checkpoint_save_path"],
                epoch,
            )
            self.save_imgs(
                fake, tir, self.cfg["train"]["train"]["img_save_path"], epoch
            )

    def save_checkpoit(self, generator, discriminator, model_save_path, epoch):
        if model_save_path is not None:
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(
                generator.state_dict(),
                model_save_path + f"epoch_{epoch}_G.pth",
            )
            torch.save(
                discriminator.state_dict(),
                model_save_path + f"epoch_{epoch}_D.pth",
            )

    def save_log(self, log):
        with open(r"./log.txt", 'a') as f:
            f.write(log)

    def save_imgs(self, fake, tir, img_save_path, epoch):
        fake_img = (
            ((fake[0].detach().squeeze(0) + 1) / 2 * 255)
            .to("cpu")
            .numpy()
            .astype(dtype=np.uint8)
        )

        tir_img = (
            ((tir[0].detach().squeeze(0) + 1) / 2 * 255)
            .to("cpu")
            .numpy()
            .astype(dtype=np.uint8)
        )
        if img_save_path is not None:
            os.makedirs(img_save_path, exist_ok=True)
            cv2.imwrite(
                img_save_path + rf"fake_{epoch}.jpg",
                fake_img,
            )
            cv2.imwrite(
                img_save_path + rf"tir_{epoch}.jpg",
                tir_img,
            )
        self.wandb_log.log(
            {
                "images/fake": wandb.Image(fake_img),
                "images/tir": wandb.Image(tir_img),
            }
        )
