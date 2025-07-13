import time
from collections import defaultdict
from tqdm import tqdm

import jittor as jt

from utils import Logger

from models.gaussian.gaussian_splatting_jittor import lpips as _lpips
from skimage.metrics import structural_similarity
from jittor.einops import reduce, rearrange, repeat

from PIL import Image
from pathlib import Path
from typing import Union
import numpy as np

from tqdm import tqdm

FloatImage = Union[
    jt.Var,  # height width or channel height width or batch channel height width
]

class DrivingForwardTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()

        self.lpips = _lpips.LPIPS(net="vgg")

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def learn(self, model):
        """
        This function sets training process.
        """        
        train_dataloader = model.train_dataloader()
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        for self.epoch in range(self.num_epochs):
                
            self.train(model, train_dataloader, start_time)
            
            # save model after each epoch using rank 0 gpu 
            if self.rank == 0:
                model.save_model(self.epoch)
                print('-'*110) 
                
            # if self.ddp_enable:
            #     dist.barrier()
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        model.set_train()
        pbar = tqdm(total=len(data_loader), desc='training on epoch {}'.format(self.epoch), mininterval=100)
        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time()
            model.optimizer.zero_grad()

            outputs, losses = model.process_batch(inputs, self.rank)
            model.optimizer.backward(losses['total_loss'])
            model.optimizer.step()

            if self.rank == 0:
                self.logger.update(
                    'train', 
                    self.epoch, 
                    self.world_size,
                    batch_idx, 
                    self.step,
                    start_time,
                    before_op_time, 
                    inputs,
                    outputs,
                    losses
                )

                if self.logger.is_checkpoint(self.step):
                    self.validate(model)

            self.step += 1
            pbar.update(1)

        pbar.close()
        model.lr_scheduler.step()
        
    @jt.no_grad()
    def validate(self, model, vis_results=False):
        """
        This function validates models on the validation dataset to monitor training process.
        """
        val_dataloader = model.val_dataloader()
        val_iter = iter(val_dataloader)
        
        # Ensure the model is in validation mode
        model.set_val()

        avg_reconstruction_metric = defaultdict(float)

        inputs = next(val_iter)
        outputs, _ = model.process_batch(inputs, self.rank)
            
        psnr, ssim, lpips= self.compute_reconstruction_metrics(inputs, outputs)

        avg_reconstruction_metric['psnr'] += psnr   
        avg_reconstruction_metric['ssim'] += ssim
        avg_reconstruction_metric['lpips'] += lpips

        print('Validation reconstruction result...\n')
        print(f"\n{inputs['token'][0]}")
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

        # Set the model back to training mode
        model.set_train()

    @jt.no_grad()
    def evaluate(self, model):
        """
        This function evaluates models on validation dataset of samples with context.
        """
        eval_dataloader = model.eval_dataloader()

        # load model
        model.load_weights()
        model.set_eval()

        avg_reconstruction_metric = defaultdict(float)

        count = 0

        process = tqdm(eval_dataloader)
        for batch_idx, inputs in enumerate(process):
            outputs, _ = model.process_batch(inputs, self.rank)
            
            psnr, ssim, lpips= self.compute_reconstruction_metrics(inputs, outputs)

            avg_reconstruction_metric['psnr'] += psnr   
            avg_reconstruction_metric['ssim'] += ssim
            avg_reconstruction_metric['lpips'] += lpips
            count += 1

            process.set_description(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}")

            print(f"\n{inputs['token'][0]}")
            print(f"avg PSNR: {avg_reconstruction_metric['psnr']/count:.4f}, avg SSIM: {avg_reconstruction_metric['ssim']/count:.4f}, avg LPIPS: {avg_reconstruction_metric['lpips']/count:.4f}")
            
        avg_reconstruction_metric['psnr'] /= len(eval_dataloader)
        avg_reconstruction_metric['ssim'] /= len(eval_dataloader)
        avg_reconstruction_metric['lpips'] /= len(eval_dataloader)

        print('Evaluation reconstruction result...\n')
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

    def save_image(
        self,
        image: FloatImage,
        path: Union[Path, str],
    ) -> None:
        """Save an image. Assumed to be in range 0-1."""

        image.float()
        # Create the parent directory if it doesn't already exist.
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Save the image.
        Image.fromarray(self.prep_image(image)).save(path)


    def prep_image(self, image: FloatImage) -> np.ndarray:  # height width channel
        # Handle batched images.
        if image.ndim == 4:
            image = rearrange(image, "b c h w -> c h (b w)")

        # Handle single-channel images.
        if image.ndim == 2:
            image = rearrange(image, "h w -> () h w")

        # Ensure that there are 3 or 4 channels.
        channel, _, _ = image.shape
        if channel == 1:
            image = repeat(image, "() h w -> c h w", c=3)
        assert image.shape[0] in (3, 4)

        image = jt.clamp(image.detach(), min_v=0, max_v=1) * 255
        return jt.uint8(rearrange(image, "c h w -> h w c")).numpy()

    @jt.no_grad()
    def compute_reconstruction_metrics(self, inputs, outputs):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0
        if self.novel_view_mode == 'SF':
            frame_id = 1
        elif self.novel_view_mode == 'MF':
            frame_id = 0
        else:
            raise ValueError(f"Invalid novel view mode: {self.novel_view_mode}")
        for cam in range(self.num_cams):
            rgb_gt = inputs[('color', frame_id, 0)][:, cam, ...]
            image = outputs[('cam', cam)][('gaussian_color', frame_id, 0)]
            psnr += self.compute_psnr(rgb_gt, image).mean()
            ssim += self.compute_ssim(rgb_gt, image).mean()
            lpips += self.compute_lpips(rgb_gt, image).mean()
            if self.save_images:
                assert self.eval_batch_size == 1
                if self.novel_view_mode == 'SF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', 0, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_0_gt.png")
                elif self.novel_view_mode == 'MF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', -1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_prev_gt.png")
                    self.save_image(inputs[('color', 1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_next_gt.png")
        psnr /= self.num_cams
        ssim /= self.num_cams
        lpips /= self.num_cams
        return psnr, ssim, lpips
    
    @jt.no_grad()
    def compute_psnr(
        self,
        ground_truth: jt.Var,  # batch channel height width
        predicted: jt.Var,  # batch channel height width
    ) -> jt.Var:  # batch
        ground_truth = ground_truth
        predicted = predicted
        ground_truth = jt.clamp(ground_truth, min_v=0, max_v=1)
        predicted = jt.clamp(predicted, min_v=0, max_v=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        base = jt.full_like(mse, 10.0)
        mse_log10 = mse.log() / base.log()
        return -10 * mse_log10
    
    @jt.no_grad()
    def compute_lpips(
        self,
        ground_truth: jt.Var,  # batch channel height width
        predicted: jt.Var,  # batch channel height width
    ) -> jt.Var:  # batch
        ground_truth = ground_truth
        predicted = predicted
        value = self.lpips.execute(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]
    
    @jt.no_grad()
    def compute_ssim(
        self,
        ground_truth: jt.Var,  # batch channel height width
        predicted: jt.Var,  # batch channel height width
    ) -> jt.Var:  # batch
        ground_truth = ground_truth
        predicted = predicted
        ssim = [
            structural_similarity(
                gt.detach().numpy(),
                hat.detach().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return jt.Var(ssim)
