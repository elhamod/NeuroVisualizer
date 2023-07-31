import torch
import torch.nn as nn
import torch.nn.functional as F

from aux.utils import get_closest_point_and_distance

####### LOSSES ###########


def rec_loss_function(recon_x, x, z):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') 
    return recon_loss 


def loss_anchor(z, predefined_values):
    l = nn.functional.mse_loss(z, predefined_values, reduction='sum')
    return l


def loss_consecutive_coordinates(coordinates):
    distances = (10*coordinates[1:] - 10*coordinates[:-1]).pow(2).sum(-1)
    max_ = torch.tensor(2*torch.pi*10*0.8/(coordinates.shape[0])).to(distances).pow(2)
    return nn.functional.mse_loss(distances, max_, reduction='sum')

    



def get_scaling_input_to_latent(input_distances, latent_distances):
    return (input_distances.max() - input_distances.min()) / (latent_distances.max() - latent_distances.min()).detach()



def loss_function_scaled_distances(input, latent):
    # Compute the pairwise distances between the input and output tensors
    input_distances = torch.cdist(input, input) #torch.cdist pairwise_distances
    latent_distances = torch.cdist(latent, latent) #torch.cdist

    # Compute the scaling factor for the latent space distances
    scaling_factor = get_scaling_input_to_latent(input_distances, latent_distances)
    
    # Scale the output distances by the scaling factor
    scaled_latent_distances = latent_distances *scaling_factor #* (scaling_factor**2) #NOTE: can be any function
    
    # Compute the mean squared error between the scaled output distances and the input distances
    mse_latent_loss = F.mse_loss(scaled_latent_distances, input_distances, reduction='mean') # 'mean'
    
    # Compute the total loss as a weighted sum of the mean squared error and the latent loss
    total_loss = mse_latent_loss 
    
    return total_loss

def loss_grid_to_trajectory(model, data_grid_latent, data_trajectory, d_max_inputspace, mode="scaled", ratio=None, latentfactor=2**2, epoch=-1):
    factor = 1
    
    _, data_trajectory_latent = model(data_trajectory)
    data_trajectory_latent = data_trajectory_latent.detach() # NOTE: we only want grid points to affect, not trajectory points
    closest_trajectory_latent_points, closest_trajectory_latent_points_index, _ = get_closest_point_and_distance(data_grid_latent, data_trajectory_latent)
    data_grid_rec = model.decoder(data_grid_latent)

    d_inputspace = torch.sqrt((data_grid_rec - data_trajectory[closest_trajectory_latent_points_index]).pow(2).sum(dim=-1))

    d_latentspace =  torch.sqrt((factor*data_grid_latent- factor*closest_trajectory_latent_points).pow(2).sum(dim=-1))

    log_dist_ratio = (torch.log(d_inputspace)/2)**2 - d_latentspace

    if ratio is None:
        max_ = torch.log(d_max_inputspace) - (factor*latentfactor)
        if epoch == 0:
            print("loss_grid_to_trajectory: Automatic ration calculated: " +str(max_.item())) #~7
            print('max param space dist^2: ', torch.log(d_max_inputspace).item())
    else:
        max_ = torch.tensor(ratio).to(log_dist_ratio)

    loss = nn.functional.mse_loss(log_dist_ratio, max_, reduction='sum') 

    return loss





