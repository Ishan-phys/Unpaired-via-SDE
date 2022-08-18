import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import random

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from models.ema import ExponentialMovingAverage
from sampling import get_sampler

class Step_by_Step(object):
    def __init__(self, sde, models, loss_fn, optimizers, config):
        """Class to train and save the model

        Args:
            sde: An `sde_lib.SDE` object that represents the forward SDE.
            models: A tuple of score models.
            loss_fn: the defined loss function
            optimizers: A tuple of optimizers to minimize the loss function
            config: configuration file
        """
        self.sde = sde
        self.models = models
        self.loss_fn = loss_fn
        self.optimizers = optimizers
        self.config = config
        
        # Set the device and send the models to the device
        self.device = config["device"]
        self.models = list(map(lambda model: torch.nn.DataParallel(model).to(self.device), self.models))

        # Set the data loaders and writer
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.total_epochs = 0
        
        # Set the states of the two models
        self.state_xy = self.get_state(self.models[0], self.optimizers[0])
        self.state_yx = self.get_state(self.models[1], self.optimizers[1])
        self.states = {"xy":self.state_xy, "yx":self.state_yx}
        
        # Internal variables
        self.losses = []
        self.val_losses = []
        
        # Set the optimizer function and the training/evaluation step function
        self.optimize_fn = self._optimization_manager()
        self.train_step_fn = self._make_train_step_fn(optimizer_fn=self.optimize_fn)
        self.val_step_fn = self._make_val_step_fn()
        
    def set_loaders(self, train_loader, val_loader=None):
        """Set the data loaders for training/evaluation.

        Args:
            train_loader: the train dataset loader
            val_loader (optional): the validation dataset loader. Defaults to None.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def set_tensorboard(self, name, folder='tensorboard'):
        """This method allows the user to define a SummaryWriter to interface with TensorBoard

        Args:
            name (str): name of the file inside the folder
            folder (str, optional): the folder where file 'name' is located. Defaults to 'tensorboard'.
        """
        if not os.path.exists(f"./{folder}"):
            os.mkdir(f"./{folder}")

        suffix = datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')
        
    def get_state(self, model, optimizer):
        ema = ExponentialMovingAverage(model.parameters(), decay=self.config["ema"]["ema_rate"])
        state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
        return state
    
    def _optimization_manager(self):
        """Returns an optimize_fn based on `config`."""

        def optimize_fn(optimizer, params, step, lr=self.config["optim"]["lr"],
                        warmup=self.config["optim"]["warmup"],
                        grad_clip=self.config["optim"]["grad_clip"]):
            
            """Optimizes with warmup and gradient clipping (disabled if negative)."""
            
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)

            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

        return optimize_fn
    
    def _make_train_step_fn(self, optimizer_fn):
        """Builds function that performs a step in the training loop

        Args:
            optimizer_fn:
        """
        states = self.states
        
        def train_step_fn(batch, i):
            """Running one step of training.

            Args:
                states: a dictionary containing the state of each of the two models
                batch: A mini-batch of training data.

            Returns:
                The average loss value of the mini-batch.
            """
            model_xy, model_yx = states["xy"]["model"], states["yx"]["model"]
            optimizer_xy, optimizer_yx = states["xy"]["optimizer"], states["yx"]["optimizer"]
            model_xy.train(), model_yx.train()
            loss = self.loss_fn(model_xy, model_yx, batch)
            loss.backward()
            
            if (i+1) % 2 == 0 or (i+1) == len(self.train_loader):
                self.optimize_fn(optimizer_xy, model_xy.parameters(), 
                                step=states["xy"]["step"])
                states["xy"]["step"] += 1
                states["xy"]["ema"].update(model_xy.parameters())
                
                self.optimize_fn(optimizer_yx, model_yx.parameters(), 
                                step=states["yx"]["step"])
                states["yx"]["step"] += 1
                states["yx"]["ema"].update(model_yx.parameters())
                
                optimizer_xy.zero_grad(set_to_none=True)
                optimizer_yx.zero_grad(set_to_none=True)

            return loss

        return train_step_fn
    
    def _make_val_step_fn(self):
        """Builds function that performs a step in the validation loop"""
        
        states = self.states
        
        def perform_val_step_fn(batch, i):
            """Running one step of validation.

            Args:
                states: a dictionary containing the states of each of the two models
                batch: A mini-batch of evaluation data.

            Returns:
                The average loss value of the mini-batch.
            """
            model_xy, model_yx = states["xy"]["model"], states["yx"]["model"]
            ema_xy, ema_yx = states["xy"]["ema"], states["yx"]["ema"]
            model_xy.eval(), model_yx.eval()
            ema_xy.store(model_xy.parameters()), ema_yx.store(model_yx.parameters())
            ema_xy.copy_to(model_xy.parameters()), ema_yx.copy_to(model_yx.parameters())
            loss = self.loss_fn(model_xy, model_yx, batch)
            ema_xy.restore(model_xy.parameters())
            ema_yx.restore(model_yx.parameters())
            
            return loss

        return perform_val_step_fn

    def _mini_batch_loss(self, validation=False):
        """Calculate the loss value for the mini-batch in either training or evaluation mode

        Args:
            validation (bool, optional): Set to true while training. Defaults to False.

        Returns:
            the calculated loss value
        """
        states = self.states
        
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
            epoch_type = "Val Epoch" 
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
            epoch_type = "Train Epoch" 
            
        if data_loader is None:
            return None
            
        mini_batch_losses = []
        
        with tqdm(data_loader, unit="batch") as tepoch:
            for i, batch in enumerate(tepoch):
                tepoch.set_description(f"{epoch_type}: {self.total_epochs}")
                mini_batch_loss = torch.mean(step_fn(batch, i))
                mini_batch_losses.append(mini_batch_loss.item())

        loss = np.mean(mini_batch_losses)
        return loss
    
    def set_seed(self, seed=42):
        """Set the seed for reproducibility

        Args:
            seed (int, optional): Defaults to 42.
        """
        if seed >= 0:
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False    
            torch.manual_seed(seed)
            random.seed(seed)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True  
        
    def train(self, n_train_iters):
        """Run the training loop over n_epochs

        Args:
            n_train_iters (int): the number of training steps
        """
        initial_step = self.total_epochs
        self.set_seed()
        
        for step in range(initial_step, n_train_iters):
            
            # Keep track of the number of epochs
            self.total_epochs +=1 
            
            # Training            
            loss = self._mini_batch_loss(validation=False)        
            self.losses.append(loss)
            
            # Validation
            #with torch.no_grad():
            #    val_loss = self._mini_batch_loss(validation=True)
            #    self.val_losses.append(val_loss)
                
            # Save the checkpoints 
            checkpoint_dir = self.config["training"]["ckpt_dir"]
            if step != 0 and step % self.config["training"]["check_pt_freq"] == 0 or step == n_train_iters:
                self.save_checkpoint(checkpoint_dir)
                # Print the current time and the number of epochs
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"Epochs Completed: {self.total_epochs}")
                print(f"Current Time: {current_time}")    
                
                if self.writer:
                    scalars = {'training': loss}
                    #if val_loss is not None:
                    #    scalars.update({'validation': val_loss})
                    self.writer.add_scalars(main_tag='loss', tag_scalar_dict=scalars, 
                                            global_step=step)
                    
        
        if self.writer:
            # Closes the writer
            self.writer.close()
            
    def save_checkpoint(self, ckpt_dir):
        """Builds dictionary with all elements for resuming training

        Args:
            ckpt_dir (str): directory where the checkpoint file is located
        """
        
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
            filepath = os.path.join(ckpt_dir, "checkpoint.pth")
            with open(filepath, 'w') as fp:
                pass
            
        states = self.states
        saved_state = {
                    'model_xy_state_dict':  states["xy"]["model"].state_dict(),
                    'optimizer_xy_state_dict':  states["xy"]["optimizer"].state_dict(),
                    'ema_xy_state_dict':    states["xy"]["ema"].state_dict(),
                    'step_xy':  states["xy"]["step"],
                    'model_yx_state_dict':  states["yx"]["model"].state_dict(),
                    'optimizer_yx_state_dict':  states["yx"]["optimizer"].state_dict(),
                    'ema_yx_state_dict': states["yx"]["ema"].state_dict(),
                    'step_yx':  states["yx"]["step"],
                    'loss': self.losses,
                    'val_loss': self.val_losses,
                    'total_epochs': self.total_epochs,
                    }
        
        filepath = os.path.join(ckpt_dir, "checkpoint.pth")
        torch.save(saved_state, filepath)

    def load_checkpoint(self, filepath):
        """Loads dictionary

        Args:
            filepath (str): directory where the checkpoint file is located
        """
        loaded_states = torch.load(filepath)

        # Restore states for models and optimizers
        self.states['xy']['model'].load_state_dict(loaded_states['model_xy_state_dict'])
        self.states['xy']['optimizer'].load_state_dict(loaded_states['optimizer_xy_state_dict'])
        self.states['xy']['ema'].load_state_dict(loaded_states['ema_xy_state_dict'])
        self.states['xy']['step'] = loaded_states['step_xy']
        
        self.states['yx']['model'].load_state_dict(loaded_states['model_yx_state_dict'])
        self.states['yx']['optimizer'].load_state_dict(loaded_states['optimizer_yx_state_dict'])
        self.states['yx']['ema'].load_state_dict(loaded_states['ema_yx_state_dict'])
        self.states['yx']['step'] = loaded_states['step_yx']
        
        self.total_epochs = loaded_states['total_epochs']
        self.losses = loaded_states['loss']
        self.val_losses = loaded_states['val_loss']
        
    def translate(self, target_domain, condition, num_steps):
        """translates a given batch of image to another domain.

        Args:
            target_domain (str): specify the domain you want to translate to
            condition: the batch of images to translate
            num_steps: the number of steps for the sampler

        Returns:
            a batch of sample images
        """
        condition = condition.to(self.device)
        sampling_fn = get_sampler(sde=self.sde, shape=condition.shape)
        if target_domain == "x":
            model = self.states["xy"]["model"]
        elif target_domain == "y":
            model = self.states["yx"]["model"]
        
        # Generate the samples
        model.eval()
        with torch.no_grad():
            samples = sampling_fn(model, condition, num_steps=num_steps)
        
        samples = samples.detach().cpu()
        
        samples_dir = self.config["sampling"]["sample_dir"]
        if not os.path.exists(samples_dir):
            os.mkdir(samples_dir)
        
        # Save the images in the samples directory
        save_image(samples, f"{samples_dir}/samples_{target_domain}.jpg")
        
        return samples 
    
    def plot_samples(self, samples):
        """Plot the batch of samples.

        Args:
            samples: a mini_batch of samples to plot
        """
        plt.figure(figsize=(4, 4))
        grid = make_grid(samples)
        np_grid = grid.numpy().transpose((1, 2, 0))
        plt.imshow(np_grid*np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5]))
        plt.axis("off")
        
    def plot_losses(self):
        """Plot the training and the validation losses."""
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()