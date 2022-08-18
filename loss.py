import torch
from sampling import get_sampler
from models.utils import get_score_fn
from configs.config import CFGS

LAMBDA = CFGS["Cycle_loss"]

def get_sde_loss_fn(sde, reduce_mean=True, continuous=True, eps=1e-5):
    
    """Create a loss function for training with arbirary SDEs.
    
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        eps: A `float` number. The smallest time step to sample from.
        
    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    device = CFGS["device"]
    
    def loss_score(t, z):
        
        def loss_scr(model, img, cond):
            """Evaluates the loss of a score function.

            Args:
                model: a score model
                img: a mini-batch of images
                cond: the conditioning

            Returns:
                evaluated loss
            """
            score_fn = get_score_fn(sde, model, continuous=continuous)
            mean_img, std_img = sde.marginal_prob(img, t)
            perturbed_img = mean_img + std_img[:, None, None, None] * z
            score = score_fn(torch.cat((perturbed_img, cond), dim=1), t)
            losses = torch.square(score * std_img[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            
            return torch.mean(losses)
        
        return loss_scr
    
    def loss_cycle(criterion):
        """Setup the cycle consistency loss.
        
        Args:
            criterion: L1 loss or the L2 loss.
        
        Returns:
            a loss function
        
        """
        if criterion == "L1":
            return torch.nn.L1Loss()
        elif criterion == "L2":
            return torch.nn.MSELoss()

    def loss_fn(model_xy, model_yx, batch):
        """Compute the loss function.
        
        Args:
            model: A score model.
            batch: A mini-batch of training data.
        
        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        
        # Load the x and y unpaired batch of dataset.
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        
        # Use the t and z to perturb the images.
        t = torch.rand(x.shape[0], device=device) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        
        # Setup the cycle consistency loss and the score loss.
        loss_cc = loss_cycle("L2")
        loss_sc = loss_score(t, z)
        
        # Initialize the samplers
        em_sampler_x = get_sampler(sde, shape=x.shape, eps=eps)
        em_sampler_y = get_sampler(sde, shape=y.shape, eps=eps)
        
        # Generate the corresponding images, x -> y_dash and y -> x_dash. 
        x_dash = em_sampler_x(model_xy, y)
        y_dash = em_sampler_y(model_yx, x)
        
        # Generate the reconstructed x_r given y_dash and y_r given x_dash.
        x_r = em_sampler_x(model_xy, y_dash)
        y_r = em_sampler_y(model_yx, x_dash)
        
        # Calculate the losses. 
        loss_1 = loss_sc(model_xy, x_dash, y)
        loss_2 = loss_sc(model_xy, x, y_dash)
        loss_3 = loss_sc(model_yx, y_dash, x)
        loss_4 = loss_sc(model_yx, y, x_dash)
        loss_cycle_x = loss_cc(x_r, x) 
        loss_cycle_y = loss_cc(y_r, y)
        
        loss = loss_1 + loss_2 + loss_3 + loss_4 + LAMBDA*loss_cycle_x + LAMBDA*loss_cycle_y
        
        return loss

    return loss_fn