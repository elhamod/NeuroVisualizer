import argparse
import csv
import joblib
import json
import pandas as pd
import torch 
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker


from aux.trajectories_data import get_trajectory_dataloader
from aux.utils import get_files, repopulate_model
from CoPhy.DNN import get_DNN
from CoPhy.lossCalculator import Loss

#A dicitionary of used methods and whether they allow inverse transforms.
modeltypes_considered = ["Kernel-PCA", "UMAP"]
NEIGHBORS=5
SEED=42

np.random.seed(SEED)

BATCH_SIZE=32
NJOBS=4

def inverse_transform(new_coordinates, best_model, method):
    coordinates_rec = best_model.inverse_transform(new_coordinates)

    print("model inverse fit!")
    return coordinates_rec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoPhy baselines training and plotting')

    #data
    parser.add_argument('--num_models', type=int, default=None, help='including n first models from the folder')
    parser.add_argument('--from_last', dest='from_last', action='store_true')
    parser.add_argument('--prefix', default='model-', help='prefix for the checkpint model')
    parser.add_argument('--model_folder', default='', help='trajectory models')
    parser.add_argument('--loss_name', '-l', default='train_loss', help='train_loss or other')
    parser.add_argument('--every_nth', type=int, default=10, help='every nth model is taken into account')

    #Cophy
    parser.add_argument('--DNN_type', type=str, help='NN or PGNN')
    # DNN_NN="NN"
    # DNN_COPHY="PGNN"
    parser.add_argument('--n_spins', type=int, default=4, help='nspin')
    parser.add_argument('--trainingCount', type=int, default=2000, help='trainingCount')
    parser.add_argument('--validation_count', type=int, default=2000, help='validation_count')
    parser.add_argument('--dataPath', type=str, help='CMT data')


    #AE archi
    parser.add_argument('--model_file', default='', help='AE model')

    #All
    parser.add_argument('--whichloss', default=None, type=str, help='whichloss to plot')

    #grid
    parser.add_argument('--xnum', type=int, default=25)
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')

    parser.add_argument('--key_models', nargs='+', help='index of key models')
    parser.add_argument('--key_modelnames', nargs='+', help='name of each')

    parser.add_argument('--density_type', type=str, default="inverse")
    parser.add_argument('--density_p', type=int, default=2)

    parser.add_argument('--density_vmax', type=float, default=-1)
    parser.add_argument('--density_vmin', type=float, default=-1)
    




    
    




    args = parser.parse_args()


    best_model_path = args.model_file
    file_path = args.model_folder
    loss_type = args.whichloss
    
    device = torch.device("cpu")


    ############ HYPERP #####


    best_model_path_directory = os.path.dirname(best_model_path)
    if not os.path.exists(best_model_path_directory):
        os.makedirs(best_model_path_directory)

    # Convert args to JSON format
    args_dict = vars(args)  # Convert Namespace object to dictionary
    json_str = json.dumps(args_dict, indent=4)  # Convert dictionary to JSON string
    # Save JSON to file
    with open(os.path.join(best_model_path_directory, 'plotting_args.json'), 'w') as f:
        f.write(json_str)


    ####!Cophy stuff
    ####Loss
    loss_obj = Loss(args.DNN_type, args.dataPath, args.n_spins, args.trainingCount, args.validation_count, device)
    D_in  = loss_obj.datasetLoader.x_dim
    D_out = loss_obj.datasetLoader.y_dim
    H = 100 # Model width
    Depth = 3 # Model depth

        
    # get files
    pt_files = get_files(file_path, args.num_models, prefix=args.prefix, from_last=args.from_last, every_nth=args.every_nth)


    rec_data_loader, transform = get_trajectory_dataloader(pt_files, BATCH_SIZE, best_model_path_directory)
    trajectory_data_loader = rec_data_loader
    dataset = rec_data_loader.dataset
    input_dim = dataset[0].shape[0]
    print('number of models considered: ', len(dataset))


    # get numpy dataset:
    samples = []
    # Iterate over the dataset and convert each sample to NumPy array
    for i in range(len(dataset)):
        sample = dataset[i]
        # Assuming each sample is a tensor, convert it to NumPy array using .numpy()
        sample_np = sample.numpy()
        samples.append(sample_np)
    # Convert the list of samples to a NumPy array
    dataset_numpy = np.array(samples)



    for method in modeltypes_considered:
        print(method, "...")
        method_path = os.path.join(best_model_path, method)
        if not os.path.exists(method_path):
            os.makedirs(method_path)
        file_path =os.path.join(method_path, "model.pkl")


        if method == "Kernel-PCA":
            from sklearn.decomposition import KernelPCA
            best_model = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True, n_jobs=NJOBS, random_state=SEED)
        elif method == "UMAP":
            from umap import UMAP
            best_model = UMAP(n_components=2, n_neighbors=NEIGHBORS)
        else:
            print(method, "is not a supported method. skipping...")
            continue
        



        


        dataset_numpy_embedded = best_model.fit_transform(dataset_numpy)
        print("model built and fit!")
        
        joblib.dump(best_model, file_path)
        trajectory_coordinates = dataset_numpy_embedded
        original_models = dataset_numpy
        original_models = original_models*transform.std.numpy() + transform.mean.numpy()
        original_models = torch.from_numpy(original_models)

        xmin_map = min(trajectory_coordinates[:, 0])/1.1 if min(trajectory_coordinates[:, 0])>0 else min(trajectory_coordinates[:, 0])*1.1
        xmax_map = max(trajectory_coordinates[:, 0])*1.1 if max(trajectory_coordinates[:, 0])>0 else max(trajectory_coordinates[:, 0])/1.1
        ymin_map = min(trajectory_coordinates[:, 1])/1.1 if min(trajectory_coordinates[:, 1])>0 else min(trajectory_coordinates[:, 1])*1.1
        ymax_map = max(trajectory_coordinates[:, 1])*1.1 if max(trajectory_coordinates[:, 1])>0 else max(trajectory_coordinates[:, 1])/1.1
        print(f"Auto calculated: [xmin, xmax] = [{xmin_map}, {xmax_map}]" )
        print(f"Auto calculated: [ymin, ymax] = [{ymin_map}, {ymax_map}]" )

        
        original_trajectory_losses = []
        for i in range(original_models.shape[0]):
            model_flattened = original_models[i, :]
            model_repopulated = repopulate_model(model_flattened, get_DNN(D_in, H, D_out, Depth, device))
            model_repopulated.eval()
            loss =  loss_obj.get_loss(model_repopulated, args.loss_name, args.whichloss).detach()
            original_trajectory_losses.append(loss)
        original_trajectory_losses = torch.stack(original_trajectory_losses)

        # scan the unit plane from 0-1 for 2D. For each step, evalute the coordinate through the decoder and get the parameters and then get the loss.
        min_x, max_x = xmin_map, xmax_map
        min_y, max_y = ymin_map, ymax_map

        step_size = min((xmax_map - xmin_map)/args.xnum, (ymax_map - ymin_map)/args.xnum)

        x_coords = torch.arange(min_x, max_x+step_size, step_size)
        y_coords = torch.arange(min_y, max_y+step_size, step_size)

        xx, yy = torch.meshgrid(x_coords, y_coords) 


        trajectory_coordinates_rec = inverse_transform(trajectory_coordinates, best_model, method)


        trajectory_models = trajectory_coordinates_rec
        trajectory_models = trajectory_models*transform.std.numpy() + transform.mean.numpy()
        trajectory_models = torch.from_numpy(trajectory_models)
        
        trajectory_losses = []
        for i in range(trajectory_models.shape[0]):
            model_flattened = trajectory_models[i, :]
            model_repopulated = repopulate_model(model_flattened, get_DNN(D_in, H, D_out, Depth, device))
            model_repopulated.eval()
            loss =  loss_obj.get_loss(model_repopulated, args.loss_name, args.whichloss).detach()
            trajectory_losses.append(loss)
        trajectory_losses = torch.stack(trajectory_losses)

    
        grid_coords = torch.stack((xx.flatten(), yy.flatten()), dim=1).numpy()

        rec_grid_models = inverse_transform(grid_coords, best_model, method)
        rec_grid_models = rec_grid_models*transform.std.numpy() + transform.mean.numpy()
        rec_grid_models = torch.from_numpy(rec_grid_models)

        grid_losses = []
        for i in range(rec_grid_models.shape[0]):
            model_flattened = rec_grid_models[i, :]
            model_repopulated = repopulate_model( model_flattened, get_DNN(D_in, H, D_out, Depth, device))
            model_repopulated.eval()
            loss = loss_obj.get_loss(model_repopulated, args.loss_name, args.whichloss).detach()
            grid_losses.append(loss)
        grid_losses = torch.stack(grid_losses)
        grid_losses = grid_losses.view(xx.shape)
            
        
        
        
        vmax = args.vmax
        vmin = args.vmin
        if args.vmax <= 0 or args.vmin <= 0:
            if args.vmax <= 0:
                vmax = max(torch.max(grid_losses).detach().cpu().numpy(), torch.max(original_trajectory_losses).detach().cpu().numpy())
                vmax = vmax*1.1
            if args.vmin <= 0:
                vmin = min(torch.min(grid_losses).detach().cpu().numpy(),torch.min(original_trajectory_losses).detach().cpu().numpy())
                vmin = vmin/1.1

            print(f"Auto calculated: [vmin, vmax] = [{vmin}, {vmax}]" )
            
            
        ######### Plotting
        levels = np.logspace(np.log10(vmin), np.log10(vmax), int(args.vlevel))


        plots_ = ['loss', 'relative_error', 'abs_error', 'dists_param_space']
        df = pd.DataFrame(columns=['index', 'file', 'x', 'y'] + plots_)

        relative_errors = (torch.abs(trajectory_losses-original_trajectory_losses)/original_trajectory_losses)
        abs_errors = (torch.abs(trajectory_losses-original_trajectory_losses))
        ds = []
            
        z = dataset_numpy_embedded
        x_recon = inverse_transform(z, best_model, method)
        transform = dataset.transform
        
        for batch_idx in range(z.shape[0]):
            row = {
                'index': batch_idx, 
                'file': os.path.basename(dataset.file_paths[batch_idx]), 
                'x': z[batch_idx, 0], 
                'y': z[batch_idx, 1], 
                'original_loss': original_trajectory_losses[batch_idx].item(),
            }
            
            x_recon_unnormalized = torch.from_numpy(x_recon[batch_idx])*transform.std + transform.mean
            data_unnormalized = torch.from_numpy(dataset_numpy[batch_idx])*transform.std+ transform.mean
            d = (data_unnormalized - x_recon_unnormalized).pow(2).sum().sqrt()
            ds.append(d)
            row['dists_param_space'] = d.item()
            row['loss'] = trajectory_losses[batch_idx].item()
            row['relative_error'] = relative_errors[batch_idx].item()
            row['abs_error'] = abs_errors[batch_idx].item()


                    
            df = df.append(row, ignore_index=True)
        
        # Calculate the mean of the specified columns
        mean_row = pd.DataFrame(df[['abs_error', 'relative_error', 'dists_param_space']].mean(axis=0)).T
        # Add a column to the mean row with the string 'Mean'
        mean_row['index'] = 'Mean'
        # Append the mean row to the original DataFrame
        df = df.append(mean_row, ignore_index=True)
        ds = torch.stack(ds)

        df.to_csv(os.path.join(method_path, 'summary_'+args.loss_name+'.csv'), quoting=csv.QUOTE_NONNUMERIC, index=False)
        
        name_map = {
                'loss': 'loss value',
                'relative_error': 'relative loss error',
                'abs_error': 'absolute loss error',
                'dists_param_space': 'projection error in parameter space',
            }
        

        #heat plot:
        import matplotlib
        fig = plt.figure()
        norm=LogNorm()

        for plot_ in plots_:
            fig = plt.figure()

            ax = plt.gca()
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

            norm=LogNorm()

        
            CS = plt.contour(xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), grid_losses.detach().cpu().numpy(), levels=levels, norm =norm)
            fmt= ticker.FormatStrFormatter('%.2e')
            ax.clabel(CS, CS.levels, fmt=lambda x: fmt(x), inline=1, fontsize=7)
                    


            if plot_ == 'relative_error':
                c = relative_errors.detach().cpu().numpy()
                cmap = None
            elif plot_ == 'abs_error':
                c = abs_errors.detach().cpu().numpy()
                cmap = None
            elif plot_ == 'loss':
                c = original_trajectory_losses.detach().cpu().numpy()  
                cmap = CS.cmap
            elif plot_ == 'dists_param_space':
                c = ds.detach().cpu().numpy()
                cmap = None
            else:
                raise "Unknown polt type"

            scatter = plt.scatter(trajectory_coordinates[:, 0], trajectory_coordinates[:, 1], c=c, marker='o', s=9, norm=norm, cmap=cmap, zorder=100)

            if hasattr(args, 'key_models') and args.key_models is not None:
                for i, idx in enumerate(args.key_models):
                    key_model_indx = int(idx)
                    key_modelname = args.key_modelnames[i]
                    plt.scatter(trajectory_coordinates[:, 0][key_model_indx], trajectory_coordinates[:, 1][key_model_indx], c=c[0], marker='o', s=8, norm=norm, edgecolors='k', cmap=cmap, zorder=100, linewidths=2) 

                    if hasattr(args, 'key_modelnames') and args.key_modelnames is not None:
                        if i == len(args.key_models)-1:
                            last_key_model_indx = trajectory_coordinates.shape[0]-1
                        else:
                            last_key_model_indx = int(args.key_models[i+1])-1
                        plt.text(trajectory_coordinates[:, 0][last_key_model_indx], trajectory_coordinates[:, 1][last_key_model_indx], key_modelname, ha='left', va='top', zorder=101, fontsize=9,backgroundcolor=(1.0, 1.0, 1.0, 0.5))
            



            # Connect the dots with lines
            if not hasattr(args, 'key_models') or args.key_models is None:
                x = trajectory_coordinates[:, 0]
                y = trajectory_coordinates[:, 1]
                for i in range(len(x)-1):
                    plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color='k')
            else:
                n=0
                for j, idx in enumerate(args.key_models):
                    if j == len(args.key_models)-1:
                        key_model_indx = trajectory_coordinates.shape[0]
                    else:
                        key_model_indx = int(args.key_models[j+1])

                    x = trajectory_coordinates[:, 0][n:key_model_indx]
                    y = trajectory_coordinates[:, 1][n:key_model_indx]
                    for i in range(len(x)-1):
                        plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color='k')
                    n = key_model_indx

            cbar = plt.colorbar(scatter, shrink=0.6)
            cbar.ax.set_ylabel( name_map[plot_]  )
            fig.savefig(os.path.join(method_path, 'map_'+loss_type+'_'+args.loss_name+'_'+plot_+'.pdf'), dpi=300, bbox_inches='tight', format='pdf')

            fig.show()



                    







                
            