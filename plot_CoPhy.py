import argparse
import csv
import json
import pandas as pd
import torch 
from AEmodel import UniformAutoencoder
from trajectories_data import get_trajectory_dataloader
import sys
import os
import numpy as np
from utils import get_density, get_files, repopulate_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoPhy plotting')

    #Cophy
    parser.add_argument('--DNN_type', type=str, help='NN or PGNN')
    # DNN_NN="NN"
    # DNN_COPHY="PGNN"
    parser.add_argument('--n_spins', type=int, default=4, help='nspin')
    parser.add_argument('--trainingCount', type=int, default=2000, help='trainingCount')
    parser.add_argument('--validation_count', type=int, default=2000, help='validation_count')
    parser.add_argument('--dataPath', type=str, help='CMT data')

    #All
    parser.add_argument('--whichloss', default=None, type=str, help='whichloss to plot')
    # parser.add_argument('--task_name', required=True, nargs='+', help='List of all task_names')

    #AE archi
    parser.add_argument('--num_of_layers', type=int, default=3)
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('--model_file', default='', help='AE model')

    #data
    parser.add_argument('--num_models', type=int, default=None, help='including n first models from the folder')
    parser.add_argument('--from_last', dest='from_last', action='store_true')
    parser.add_argument('--prefix', default='model-', help='prefix for the checkpint model')
    parser.add_argument('--model_folder', default='', help='trajectory models')
    parser.add_argument('--loss_name', '-l', default='train_loss', help='train_loss or other')
    parser.add_argument('--every_nth', type=int, default=10, help='every nth model is taken into account')

    #grid
    parser.add_argument('--x', default='-1:1:25', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')

    parser.add_argument('--key_models', nargs='+', help='index of key models')
    parser.add_argument('--key_modelnames', nargs='+', help='name of each')

    parser.add_argument('--density_type', type=str, default="inverse")
    parser.add_argument('--density_p', type=int, default=2)

    parser.add_argument('--density_vmax', type=float, default=-1)
    parser.add_argument('--density_vmin', type=float, default=-1)

    latent_dim = 2



    args = parser.parse_args()

    min_map, max_map, xnum = [float(a) for a in args.x.split(':')]
    step_size = (max_map - min_map)/xnum

    best_model_path = args.model_file
    file_path = args.model_folder
    loss_type = args.whichloss
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ##Cophy stuff##### #NOTE: replace these for a different mode type
    Convectionpath ='../CoPhy/'
    if os.path.exists(Convectionpath) and Convectionpath not in sys.path:
            sys.path.insert(0, Convectionpath)
    from DNN import get_DNN
    from lossCalculator import Loss


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



    #### Data
    pt_files = get_files(file_path, args.num_models, prefix=args.prefix, from_last=args.from_last, every_nth=args.every_nth)

    trajectory_data_loader, transform = get_trajectory_dataloader(pt_files, args.batch_size, best_model_path_directory)
    trajectory_dataset = trajectory_data_loader.dataset
    input_dim = trajectory_dataset[0].shape[0]





    ##########MODEL

    # load the model
    best_model = UniformAutoencoder(input_dim, args.num_of_layers, latent_dim).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model = best_model.to(device)
    best_model.eval()








    ###### Get coordinates and losses of trajectories
    trajectory_coordinates = []
    trajectory_dataset_samples = []
    trajectory_coordinates_rec = []
    with torch.no_grad():
        for batch_idx, data in enumerate(trajectory_dataset):
            data = data.to(device).view(1, -1)

            x_recon, z = best_model(data)

            trajectory_coordinates.append(z)
            trajectory_coordinates_rec.append(x_recon)
						   
            trajectory_dataset_samples.append(data)
    trajectory_coordinates = torch.cat(trajectory_coordinates, dim=0)
    trajectory_models = torch.cat(trajectory_coordinates_rec, dim=0)
    original_models = torch.cat(trajectory_dataset_samples, dim=0)

    trajectory_models = trajectory_models*transform.std.to(device) + transform.mean.to(device) 
    original_models = original_models*transform.std.to(device) + transform.mean.to(device)



    trajectory_losses = []
    for i in range(trajectory_models.shape[0]):
        model_flattened = trajectory_models[i, :]
        model_repopulated = repopulate_model(model_flattened, get_DNN(D_in, H, D_out, Depth, device))
        model_repopulated.eval()
        model_repopulated = model_repopulated.to(device)
        loss =  loss_obj.get_loss(model_repopulated, args.loss_name, args.whichloss).detach()
        trajectory_losses.append(loss)
    trajectory_losses = torch.stack(trajectory_losses)


    original_trajectory_losses = []
    for i in range(original_models.shape[0]):
        model_flattened = original_models[i, :]
        model_repopulated = repopulate_model(model_flattened, get_DNN(D_in, H, D_out, Depth, device))
        model_repopulated.eval()
        model_repopulated = model_repopulated.to(device)
        loss =  loss_obj.get_loss(model_repopulated, args.loss_name, args.whichloss).detach()
        original_trajectory_losses.append(loss)
    original_trajectory_losses = torch.stack(original_trajectory_losses)

        





    ###### Get coordinates and losses of surface

    # scan the unit plane from 0-1 for 2D. For each step, evalute the coordinate through the decoder and get the parameters and then get the loss.
    min_x, max_x = min_map, max_map
    min_y, max_y = min_map, max_map

    x_coords = torch.arange(min_x, max_x+step_size, step_size)
    y_coords = torch.arange(min_y, max_y+step_size, step_size)

    xx, yy = torch.meshgrid(x_coords, y_coords)
    grid_coords = torch.stack((xx.flatten(), yy.flatten()), dim=1).to(device)

    rec_grid_models = best_model.decoder(grid_coords)
    rec_grid_models = rec_grid_models*transform.std.to(device) + transform.mean.to(device)

    grid_losses = []
    for i in range(rec_grid_models.shape[0]):
        model_flattened = rec_grid_models[i, :]
        model_repopulated = repopulate_model( model_flattened, get_DNN(D_in, H, D_out, Depth, device))
        model_repopulated.eval()
        model_repopulated = model_repopulated.to(device)
        loss = loss_obj.get_loss(model_repopulated, args.loss_name, args.whichloss).detach()
        grid_losses.append(loss)
    grid_losses = torch.stack(grid_losses)
    grid_losses = grid_losses.view(xx.shape)



    vmax = args.vmax
    vmin = args.vmin
    if args.vmax <= 0:
        vmax = max(torch.max(grid_losses).detach().cpu().numpy(), torch.max(original_trajectory_losses).detach().cpu().numpy())
        vmax = vmax*1.1
    if args.vmin <= 0:
        vmin = min(torch.min(grid_losses).detach().cpu().numpy(),torch.min(original_trajectory_losses).detach().cpu().numpy())
        vmin = vmin/1.1
    print(f"Auto calculated: [vmin, vmax] = [{vmin}, {vmax}]" )


    ######### Plotting

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    import matplotlib.ticker as ticker
    levels = np.logspace(np.log10(vmin), np.log10(vmax), int(args.vlevel))


    plots_ = ['loss', 'relative_error', 'abs_error', 'dists_param_space']
    df = pd.DataFrame(columns=['index', 'file', 'x', 'y'] + plots_)

    relative_errors = (torch.abs(trajectory_losses-original_trajectory_losses)/original_trajectory_losses)
    ds = []
    abs_errors = (torch.abs(trajectory_losses-original_trajectory_losses))
    ds = []
    for batch_idx, data in enumerate(trajectory_dataset):   
        data = data.to(device)

        x_recon, z = best_model(data.view(1, -1))
        z = z.view(-1)

        transform = trajectory_dataset.transform
        data_unnormalized = data*transform.std.to(device) + transform.mean.to(device)
        x_recon_unnormalized = x_recon*transform.std.to(device) + transform.mean.to(device)
        d = (data_unnormalized - x_recon_unnormalized).pow(2).sum().sqrt()
        ds.append(d)

        row = {
                'index': batch_idx, 
                'file': os.path.basename(trajectory_dataset.file_paths[batch_idx]), 
                'x': z[0].detach().cpu().numpy(), 
                'y': z[1].detach().cpu().numpy(), 
                'dists_param_space': d.item(),
                'loss': trajectory_losses[batch_idx].item(),
                'original_loss': original_trajectory_losses[batch_idx].item(),
                'relative_error': relative_errors[batch_idx].item(),
                'abs_error': abs_errors[batch_idx].item()}
                    
        df = df.append(row, ignore_index=True)

    # Calculate the mean of the specified columns
    mean_row = pd.DataFrame(df[['abs_error', 'relative_error', 'dists_param_space']].mean(axis=0)).T
    # Add a column to the mean row with the string 'Mean'
    mean_row['index'] = 'Mean'
    # Append the mean row to the original DataFrame
    df = df.append(mean_row, ignore_index=True)

    df.to_csv(os.path.join(best_model_path_directory, 'summary_'+args.loss_name+'.csv'), quoting=csv.QUOTE_NONNUMERIC, index=False)
    ds = torch.stack(ds)

    
    name_map = {
            'train_loss': 'Training',
            'test_loss': 'Test',
            'val_loss': 'Validation',
            'loss': 'loss value',
            'relative_error': 'relative loss error',
            'abs_error': 'absolute loss error',
            'dists_param_space': 'projection error in parameter space',
        }
    

    #heat plot:
    import matplotlib
    fig = plt.figure()
    norm=LogNorm()
    density = get_density(rec_grid_models.detach().cpu().numpy(), args.density_type, args.density_p)
    if args.density_vmax <= 0 or args.density_vmin <= 0:
        density_vmax = np.max(density)
        density_vmax = density_vmax*1.1
        density_vmin = np.min(density)
        density_vmin = density_vmin/1.1
        print(f"Auto calculated: [density_vmin, density_vmax] = [{density_vmin}, {density_vmax}]" )
    else:
        density_vmax = args.density_vmax
        density_vmin = args.density_vmin
        print(f"[density_vmin, density_vmax] = [{density_vmin}, {density_vmax}]" )
        
    levels_density = np.logspace(np.log10(density_vmin), np.log10(density_vmax), int(args.vlevel))
    density = density.reshape(list(xx.shape))
    CS = plt.contour(xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), density, norm =norm, levels=levels_density)
    plt.xlabel('x')
    plt.ylabel('y')
    k= matplotlib.colors.Normalize(vmin=CS.cvalues.min(), vmax=CS.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    scatter = plt.scatter(trajectory_coordinates[:, 0].detach().cpu().numpy(), trajectory_coordinates[:, 1].detach().cpu().numpy(), c='0.5', marker='o', s=8, zorder=100)
    cbar.ax.set_ylabel( "Density")
    fig.savefig(os.path.join(best_model_path_directory, 'map_'+loss_type+'_grid_density.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    fig.show()

    

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

        scatter = plt.scatter(trajectory_coordinates[:, 0].detach().cpu().numpy(), trajectory_coordinates[:, 1].detach().cpu().numpy(), c=c, marker='o', s=8, norm=norm, cmap=cmap, zorder=100)
        


        if hasattr(args, 'key_models') and args.key_models is not None:
            for i, idx in enumerate(args.key_models):
                key_model_indx = int(idx)
                key_modelname = args.key_modelnames[i]
                plt.scatter(trajectory_coordinates[:, 0][key_model_indx].detach().cpu().numpy(), trajectory_coordinates[:, 1][key_model_indx].detach().cpu().numpy(), c=c[0], marker='o', s=8, norm=norm, edgecolors='k', cmap=cmap, zorder=100, linewidths=2) 

                if hasattr(args, 'key_modelnames') and args.key_modelnames is not None:
                    if i == len(args.key_models)-1:
                        last_key_model_indx = trajectory_coordinates.shape[0]-1
                    else:
                        last_key_model_indx = int(args.key_models[i+1])-1
                    plt.text(trajectory_coordinates[:, 0][last_key_model_indx].detach().cpu().numpy(), trajectory_coordinates[:, 1][last_key_model_indx].detach().cpu().numpy(), key_modelname, ha='left', va='top', zorder=101, fontsize=9,backgroundcolor=(1.0, 1.0, 1.0, 0.5))

        



        # Connect the dots with lines
        if not hasattr(args, 'key_models') or args.key_models is None:
            x = trajectory_coordinates[:, 0].detach().cpu().numpy()
            y = trajectory_coordinates[:, 1].detach().cpu().numpy()
            for i in range(len(x)-1):
                plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color='k')
        else:
            n=0
            for j, idx in enumerate(args.key_models):
                if j == len(args.key_models)-1:
                    key_model_indx = trajectory_coordinates.shape[0]
                else:
                    key_model_indx = int(args.key_models[j+1])

                x = trajectory_coordinates[:, 0][n:key_model_indx].detach().cpu().numpy()
                y = trajectory_coordinates[:, 1][n:key_model_indx].detach().cpu().numpy()
                for i in range(len(x)-1):
                    plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color='k')
                n = key_model_indx

        cbar = plt.colorbar(scatter, shrink=0.6)
        cbar.ax.set_ylabel( name_map[plot_]  )

        fig.savefig(os.path.join(best_model_path_directory, 'map_'+loss_type+'_'+args.loss_name+'_'+plot_+'.pdf'), dpi=300, bbox_inches='tight', format='pdf')

        fig.show()


