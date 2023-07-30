import re
import torch
import os
import numpy as np
import csv

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader, Dataset

from trajectories_data import get_trajectory_dataset

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#### plotting
def plot_losses(df, every_epoch, file_path):
    plt.figure()
    
    # Set x and y axis labels
    x_label = 'Epoch'
    y_label = 'Loss'    

    # Set plot style
    plt.style.use('seaborn-whitegrid')

    # Create plot
    fig, ax = plt.subplots()

    # Plot each column as a line plot
    for column in df.columns:
        if column != 'epoch':
            ax.plot(df['epoch'], df[column], label=column)

    # Set x and y axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set x-axis tick intervals
    ax.set_xticks(df['epoch'][::every_epoch])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    # Set log scale on y-axis
    plt.yscale('log')
    # plt.ylim(bottom=0.1)

    # Set legend
    ax.legend()

    # Save plot to PNG file with 300 dpi
    plt.savefig(os.path.join(file_path, 'losses.pdf'), dpi=300)

    # Show plot
    plt.show()

    plt.close()
    plt.close(fig)


####### stats stuff #####

def print_stats(data_loader, filepath, postfix=""):#, unnormalize=False):
    min_val = float('inf')
    max_val = float('-inf')
    sum_val = 0
    sum_squared_val = 0
    count = 0

    for batch in data_loader:
        data = batch.float()
        min_val = min(min_val, torch.min(data))
        max_val = max(max_val, torch.max(data))
        sum_val += torch.sum(data)
        sum_squared_val += torch.sum(data ** 2)
        count += data.numel()

    mean_val = sum_val / count
    std_val = torch.sqrt((sum_squared_val / count) - (mean_val ** 2))

    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")
    print(f"Mean value: {mean_val}")
    print(f"Standard deviation: {std_val}")

    data = {
        'Minimum value': [min_val.item()],
        'Maximum value': [max_val.item()],
        'Mean value': [mean_val.item()],
        'Standard deviation': [std_val.item()]}
    # Create pandas DataFrame from dictionary
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(filepath, 'dataset_stats'+postfix+'.csv'), quoting=csv.QUOTE_NONNUMERIC)

def print_coordinates(data_loader2, best_model, predefined_values, device, best_model_path_directory):
    df = pd.DataFrame(columns=['filename', 'x', 'y', 'x_true', 'y_true'])

    best_model.eval()
    dataset = data_loader2.dataset.dataset #NOTE: because we use a subset
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            data = data.to(device)

            # Get the model predictions
            _, z = best_model(data)

            # Calculate the errors
            print('z,z_true = ', z, predefined_values[batch_idx])

            z_list = z.tolist()
            predefined_values_list = predefined_values.tolist()
            print(z_list, predefined_values_list)
            df = df.append({'filename': os.path.basename(dataset.file_paths[batch_idx]), 'x': z_list[0], 'y': z_list[1], 'x_true': predefined_values_list[batch_idx][0], 'y_true': predefined_values_list[batch_idx][1]}, ignore_index=True)

        df.to_csv(os.path.join(best_model_path_directory, 'coordinates.csv'), quoting=csv.QUOTE_NONNUMERIC, index=False)

def print_errors(data_loader, best_model, device, best_model_path_directory, df, epoch, err_tolerance=0.01, unnormalize=True):
    errors = []
    best_model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)

            # Get the model predictions
            x_recon, z = best_model(inputs)

            if unnormalize:
                t = data_loader.dataset.transform
                x_recon = x_recon*t.std.to(device) + t.mean.to(device)
                inputs = inputs*t.std.to(device) + t.mean.to(device)

            # Calculate the errors
            error = (x_recon - inputs).pow(2).sum(dim=1).sqrt().view(-1, 1)
            errors.append(error)

        errors_tensor = torch.cat(errors, dim=0)
        min_error = torch.min(errors_tensor).item()
        max_error = torch.max(errors_tensor).item()
        mean_error = torch.mean(errors_tensor).item()
        std_error = torch.std(errors_tensor).item()


        # Set the range to check
        range_min = -err_tolerance
        range_max = err_tolerance

        # Create a boolean mask for the values within the range
        mask = torch.logical_and(errors_tensor >= range_min, errors_tensor <= range_max)

        # Calculate the proportion of values within the range
        prob_in_range = torch.mean(mask.float()).item()

        # save the array to a CSV file
        file_prefix=os.path.join(best_model_path_directory,'errors'+('_unnormalized' if unnormalize else ''))
        # open file for writing
        print(f"Minimum error: {min_error}")
        print(f"Maximum error: {max_error}")
        print(f"Mean error: {mean_error}")
        print(f"Standard deviation of error: {std_error}")
        print(f"Probability that a value is in the range [{range_min}, {range_max}]: {prob_in_range:.2f}")

        data = {
            'epoch': epoch,
            'Minimum error': min_error,
            'Maximum error': max_error,
            'Mean error': mean_error,
            'Standard deviation of error': std_error,
            f'Probability that a value is in the range [{range_min}, {range_max}]': prob_in_range}
        # Create pandas DataFrame from dictionary
        if df is None:
            df = pd.DataFrame(data, index=['epoch'])
        else:
            df = df.append(data, ignore_index=True)
        df.to_csv(file_prefix +'.csv', quoting=csv.QUOTE_NONNUMERIC)

        return df



####### Model stuff ####
def repopulate_model(flattened_params, new_model):
    start_idx = 0
    state_dict = new_model.state_dict()
    for name, param in enumerate(state_dict):
        param = state_dict[param]
        size = param.numel()
        sub_flattened = flattened_params[start_idx : start_idx + size].view(param.size())
        param.data.copy_(sub_flattened)
        start_idx += size
    return new_model

def get_closest_point_and_distance(grid_points, trajectory_points):
    # Compute the distances between each point in tensor1 and tensor2
    distances = torch.cdist(grid_points, trajectory_points)

    # Find the index of the closest point in tensor2 for each point in tensor1
    closest_trajectory_points_index = torch.argmin(distances, dim=1)

    # Extract the closest points from tensor2 for each point in tensor1
    closest_trajectory_points = trajectory_points[closest_trajectory_points_index]

    # Compute the distances between each point in tensor1 and its closest point in tensor2
    distance_from_closest_trajectory = torch.gather(distances, 1, closest_trajectory_points_index.unsqueeze(1))

    return closest_trajectory_points, closest_trajectory_points_index, distance_from_closest_trajectory





##### Misc

def get_files(file_path, num_models=None, prefix="", from_last=False, every_nth=1):
    def extract_number(s, prefix=prefix):
        pattern = re.compile(r'{}(\d+).pt'.format(prefix))
        match = pattern.search(s)
        if match:
            return int(match.group(1))
        else:
            return float('inf')

    def get_all_files(d):
        f_ = []
        for dirpath, dirnames, filenames in os.walk(d):
            f_temp = []
            for filename in filenames:
                f_temp.append(os.path.join(dirpath, filename))

            f_temp = [file for file in f_temp if os.path.splitext(file)[-1] == ".pt"]
            f_temp = sorted(f_temp, key=extract_number)
            
            len_f_temp_original = len(f_temp)
            print(dirpath, 'has', len_f_temp_original, 'files')
            
            if every_nth>1 and len_f_temp_original>0:
                f_temp_last = f_temp[-1]
                f_temp = f_temp[::every_nth]
                
                if len_f_temp_original % every_nth != 1:
                   f_temp = f_temp +  [f_temp_last]
            
            f_ = f_ + f_temp
        return f_
    
    directory = os.path.join(file_path)
    files=get_all_files(directory)
    print(files)
    pt_files = [file for file in files if file.endswith(".pt")]
    print(len(pt_files), 'files included.')
    if num_models is not None:
        pt_files = pt_files[:num_models] if not from_last else pt_files[-num_models:]

    return pt_files


def get_gridpoint_dataset(grid_step=0.1):    
    min_grid=-1.
    max_grid=1.
    # Generate a 1D tensor of x values
    x_values = torch.arange(min_grid, max_grid, grid_step)
    # Generate a 1D tensor of y values
    y_values = torch.arange(min_grid, max_grid, grid_step)
    # Use meshgrid and stack to generate a 2D tensor of all coordinates on the grid
    X, Y = torch.meshgrid(x_values, y_values)
    grid_tensor = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    # Create a TensorDataset from the tensor
    dataset = TensorDataset(grid_tensor)
    # Create a DataLoader from the dataset
    return dataset


class PaddedConcatDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            data1 = self.dataset1[idx]
            if idx < len(self.dataset2):
                data2 = self.dataset2[idx]
            else:
                # Pad the smaller dataset with its own samples
                data2 = self.dataset2[idx % len(self.dataset2)]
        else:
            data2 = self.dataset2[idx]
            if idx < len(self.dataset1):
                data1 = self.dataset1[idx]
            else:
                # Pad the smaller dataset with its own samples
                data1 = self.dataset1[idx % len(self.dataset1)]

        return (data1, data2)

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))
    
def get_gridpoint_and_trajectory_datasets(pt_files, path, grid_step=0.1, batch_size=32):    
    dataset_gridpoint = get_gridpoint_dataset(grid_step=grid_step)
    dataset_trajectory, _ = get_trajectory_dataset(pt_files, path)
    
    concat_dataset = PaddedConcatDataset(dataset_gridpoint, dataset_trajectory)

    dataloader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=False)

    return dataloader



def get_diagonal_mask(n, offset=1):
    """
    Returns a Boolean tensor of shape (n, n) with True values in the indices next to
    the diagonal entries and False elsewhere.
    """
    diag = torch.ones(n - abs(offset), dtype=torch.bool)
    eye = torch.diag(diag, diagonal=offset)
    return eye


def loss_well_spaced_trajectory(coords):
    # Compute distances between adjacent coordinates
    prev_dists = torch.norm(coords[1:] - coords[:-1], dim=1)
    prev_dists = torch.cat([torch.tensor([0]).to(prev_dists), prev_dists], dim=0).unsqueeze(1)
    next_dists = torch.norm(coords[:-1] - coords[1:], dim=1)
    next_dists = torch.cat([next_dists, torch.tensor([0]).to(next_dists)], dim=0).unsqueeze(1)

    # Compute pair-wise distances
    pairwise_dists = torch.cdist(coords, coords)

    # Compute condition tensor
    condition_next = (pairwise_dists < next_dists)#.all(dim=0)
    condition_prev = (pairwise_dists < prev_dists)

    where_closer_next = torch.where(condition_next, next_dists - pairwise_dists, torch.tensor([0.0]).to(next_dists))
    where_closer_previous = torch.where(condition_prev, prev_dists - pairwise_dists, torch.tensor([0.0]).to(next_dists))

    # Create mask tensor
    mask_previous = get_diagonal_mask(where_closer_previous.shape[0], offset=1).to(condition_next)
    mask_next = get_diagonal_mask(where_closer_next.shape[0], offset=-1).to(condition_next)
    mask_diag = get_diagonal_mask(where_closer_next.shape[0],  offset=0).to(condition_next)
    mask = ~(mask_previous | mask_next | mask_diag)

    where_closer = torch.where(mask, where_closer_previous + where_closer_next, torch.tensor([0.0]).to(next_dists))

    loss = torch.sum(where_closer)
    
    return loss








##################################
#Density measures
##################################


def get_density(grid, type="inverse", p=1):
    if type == "inverse":
        return get_density_inverse(grid, p) 
    elif type == "cos":
        return get_density_COS(grid) 
    elif type == "CKA":
        return get_density_CKA(grid) 
    else:
        raise "density measure not implemented"



# CKA similiarity
def linear_kernel(X, Y):
    # X: n x m
    # Y: n x m
    return np.dot(X, Y.T)
    # Returns: n x n
def centered_kernel_matrix(matrix):
    # matrix: n x m
    K = linear_kernel(matrix, matrix)
    # K: n x n

    n = K.shape[0]
    ones = np.ones((n, n)) / n
    H = np.eye(n) - ones
    HKH = H @ K @ H
    # HKH: n x n

    return HKH
def get_density_CKA(matrix):
    # matrix: n x m
    HKH = centered_kernel_matrix(matrix)
    # HKH: n x n

    norms = np.sqrt(np.diag(HKH))
    # norms: n

    outer_norms = np.outer(norms, norms)
    # outer_norms: n x n

    cka_matrix = HKH / outer_norms
    cka_matrix = (cka_matrix + 1) / 2
    cka_matrix = np.clip(cka_matrix, 0 ,1)
    # cka_matrix: n x n

    density = np.sum(cka_matrix, axis=1, keepdims=True)

    return density




def get_density_COS(grid):
    normalized_grid = normalize(grid, axis=1)
    cosine_similarities = cosine_similarity(normalized_grid)
    cosine_similarities = np.abs(np.clip(cosine_similarities, -1, 1))
    weights = cosine_similarities

    density = np.sum(weights, axis=1, keepdims=True)

    return density

def get_density_inverse(grid, power=1):
    radius = None

    # calculate distances between each data point and all other data points
    distances = (grid[:, np.newaxis, :] - grid)**2
    distances = np.sum(distances, axis=2)
    distances = np.sqrt(distances)
    distances = np.where(distances == 0, np.inf, distances)
    
    # apply search radius if specified
    if radius is not None:
        distances[distances > radius] = np.inf
    
    # calculate weights based on the inverse distance
    weights = 1.0 / distances**power

    density = np.sum(weights, axis=1, keepdims=True)

    return density