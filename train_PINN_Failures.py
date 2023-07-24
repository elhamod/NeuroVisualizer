import argparse
import json
import pandas as pd
import torch
import os
import itertools
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import warnings
warnings.filterwarnings("ignore", message="The frame.append method is deprecated")


from AEmodel import UniformAutoencoder
from earlystopping import EarlyStopping
from losses import loss_grid_to_trajectory, rec_loss_function, loss_anchor
from trajectories_data import get_trajectory_dataloader, get_anchor_dataloader, get_predefined_values
from utils import get_files, get_gridpoint_and_trajectory_datasets, loss_well_spaced_trajectory, plot_losses, print_coordinates, print_errors, print_stats








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wang et. al. training')

    #AE archi
    parser.add_argument('--num_of_layers', type=int, default=3)
    parser.add_argument('--layers_AE', nargs='+', type=int, default=None) #NOTE: overrides num_of_layers
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('--epochs', default=40000, type=int, help='epochs')
    parser.add_argument('--patience_scheduler', default=40000, type=int, help='early stopping')
    parser.add_argument('--every_epoch', default=100, type=int, help='logging')
    parser.add_argument('--cosine_Scheduler_patience', default=600, type=int, help='cycle for scheduler')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='lr')
    parser.add_argument('--model_file', default='', help='AE model')
    

    #data
    parser.add_argument('--num_models', type=int, default=None, help='including n first models from the folder')
    parser.add_argument('--from_last', dest='from_last', action='store_true')
    parser.add_argument('--prefix', default='model-', help='prefix for the checkpint model')
    parser.add_argument('--model_folder', default='', help='trajectory models')
    parser.add_argument('--every_nth', type=int, default=1, help='every nth model is taken into account')

    #grid
    parser.add_argument('--grid_step', default=0.1, type=float, help='grid step for grids loss')
    parser.add_argument('--grid_ratio', default=None, type=float, help='grid rati of input to latent')
    parser.add_argument('--latentfactor', default=2, type=float, help='latentfactor') 
    parser.add_argument('--anchor_mode', default="diagonal", type=str, help='anchor shape') #"circle"#"diagonal"
    parser.add_argument('--grid_mode', default="proportional", type=str, help='grid mode') #"proportional"#"scaled"

    #weigths
    parser.add_argument('--rec_weight', default=1.0, type=float)
    parser.add_argument('--anchor_weight', default=0.0, type=float)
    parser.add_argument('--lastzero_weight', default=0.0, type=float)
    parser.add_argument('--firstzero_weight', default=0.0, type=float)
    parser.add_argument('--polars_weight', default=0.0, type=float)
    parser.add_argument('--equidistant_weight', default=0.0, type=float)
    parser.add_argument('--wellspacedtrajectory_weight', default=0.0, type=float)
    parser.add_argument('--gridscaling_weight', default=0.0, type=float)

    parser.add_argument('--resume', dest='resume', action='store_true')

    latent_dim = 2



    args = parser.parse_args()

    file_path = args.model_folder
    best_model_path = args.model_file








    ###### HYPERP ######
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_dict = {
        'rec': {
            'weight': args.rec_weight, #10.0,  # Adjust the weight of the uniform loss term
            'official_name': "Reconstruction loss"
        },
        'anchor': {
            'weight': args.anchor_weight, #1.0,
            'official_name': "Anchor loss"
        },
        'lastzero': {
            'weight': args.lastzero_weight, #100.0,
            'official_name': "LastZero loss"
        },
        'firstzero': {
            'weight': args.firstzero_weight, #100.0,
            'official_name': "FirstZero loss"
        },
        'polars': {
            'weight': args.polars_weight, #100.0,
            'official_name': "Polar loss"
        },
        'gridscaling': {
            'weight': args.gridscaling_weight, #1.0,
            'official_name': "Grid-scaling loss"
        },
        'wellspacedtrajectory': {
            'weight': args.wellspacedtrajectory_weight, #1.0,
            'official_name': "Well-spaced-trajectory loss"
        },
    }
    isEnabled = lambda loss: loss_dict[loss]['weight'] > 0


    
    best_model_path_directory = os.path.dirname(best_model_path)
    if not os.path.exists(best_model_path_directory):
        os.makedirs(best_model_path_directory)

    # Convert args to JSON format
    args_dict = vars(args)  # Convert Namespace object to dictionary
    json_str = json.dumps(args_dict, indent=4)  # Convert dictionary to JSON string
    # Save JSON to file
    with open(os.path.join(best_model_path_directory, 'args.json'), 'w') as f:
        f.write(json_str)




    # get files
    pt_files = get_files(file_path, args.num_models, prefix=args.prefix, from_last=args.from_last, every_nth=args.every_nth)

    range_of_files_for_anchor = range(len(pt_files))


    rec_data_loader, transform = get_trajectory_dataloader(pt_files, args.batch_size, best_model_path_directory)
    loss_dict['rec']['dataloader'] = rec_data_loader
    dataset = rec_data_loader.dataset
    input_dim = dataset[0].shape[0]
    print('number of models considered: ', len(dataset))


    #stats
    print('normalzied model data')
    print_stats(rec_data_loader, best_model_path_directory)
    print('unnormalzied  model data')
    data_loader_unnormalized, _ = get_trajectory_dataloader(pt_files, args.batch_size, best_model_path_directory, normalize=False)
    print_stats(data_loader_unnormalized, best_model_path_directory, 'unnormalized') 


    if isEnabled('anchor'):
        anchor_dataloader = get_anchor_dataloader(dataset, range_of_files_for_anchor)
        loss_dict['anchor']['dataloader'] = anchor_dataloader
        predefined_values = get_predefined_values(anchor_dataloader.dataset, args.anchor_mode)
        predefined_values = predefined_values.to(device)

    if isEnabled('lastzero'):
        loss_dict['lastzero']['dataloader'] =  get_anchor_dataloader(dataset)
    if isEnabled('firstzero'):
        loss_dict['firstzero']['dataloader'] =  get_anchor_dataloader(dataset)
    if isEnabled('polars'):
        loss_dict['polars']['dataloader'] =  get_anchor_dataloader(dataset)


    d_max_inputspace=None
    if isEnabled('gridscaling'):
        loss_dict['gridscaling']['dataloader'] = get_gridpoint_and_trajectory_datasets(pt_files, best_model_path_directory, args.grid_step, batch_size=args.batch_size)
        data_trajectory_dataset_temp = loss_dict['gridscaling']['dataloader'].dataset
        data_trajectory_dataset_temp_0 = data_trajectory_dataset_temp[0][1]
        data_trajectory_dataset_temp_last = data_trajectory_dataset_temp[-1][1]
        d_max_inputspace = torch.sqrt((data_trajectory_dataset_temp_0 - data_trajectory_dataset_temp_last).pow(2).sum(dim=-1)).to(device)

    if isEnabled('wellspacedtrajectory'):
        loss_dict['wellspacedtrajectory']['dataloader'], _ = get_trajectory_dataloader(pt_files, len(pt_files), best_model_path_directory, shuffle=False)


    model = UniformAutoencoder(input_dim, args.num_of_layers, latent_dim, h=args.layers_AE).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, args.cosine_Scheduler_patience)




    #### Training

    def cycle_dataloader(dataloader):
        """Returns an infinite iterator for a dataloader."""
        return itertools.cycle(iter(dataloader))

    if (not os.path.exists(best_model_path)) and (args.resume):
        raise "Can't resume without a model"
    
    best_model = None
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        best_model = model

    if (best_model is not None) and (not args.resume):
        raise "There is a model already. Use --resume to update it."
    




    earlystopping = EarlyStopping(model, best_model_path, patience=args.patience_scheduler)
    earlystopping.on_train_begin()

    columns=['epoch']
    iterators = {}
    for i in loss_dict.keys():
        if isEnabled(i):
            iterators[i] = {
                'iterator': iter(cycle_dataloader(loss_dict[i]['dataloader'])),
                'maxbatch': len(loss_dict[i]['dataloader'])
            }
            columns.append(loss_dict[i]['official_name'])

    max_batches = max([iterators[d]['maxbatch'] for d in iterators.keys()])

    
    columns.append('Total loss')
    columns.append('Learning rate')
    df_losses = pd.DataFrame(columns=columns)
    df_errors = None
    df_errors_unnormalized = None
    
    for epoch in range(args.epochs):
        # print('ho', epoch)
        if earlystopping.stop_training == False:
            model.train()
            total_losses = {}
            for i in loss_dict.keys():
                total_losses[i] = 0
            total_loss = 0
        
            for batch_idx in range(max_batches):
                optimizer.zero_grad()
                losses = {}

                data = {}
                for i in loss_dict.keys():
                    if isEnabled(i):
                        data[i] = next(iterators[i]['iterator'])


                if isEnabled('rec'):
                    data['rec'] = data['rec'].to(device)
                    x_recon, z = model(data['rec'])
                    loss_t = 0
                    losses['rec'] = rec_loss_function(x_recon, data['rec'], z)

                if isEnabled('anchor'):
                    data['anchor'] = data['anchor'].to(device)
                    x_recon, z = model(data['anchor'])
                    losses['anchor'] = loss_anchor(z, predefined_values)
                
                if isEnabled('lastzero'):
                    data['lastzero'] = data['lastzero'].to(device)
                    x_recon, z = model(data['lastzero'])
                    last_coordinate = z[-1, :]
                    loss_zero = torch.nn.functional.mse_loss(10*last_coordinate, torch.zeros_like(last_coordinate))
                    losses['lastzero'] = loss_zero
                if isEnabled('firstzero'):
                    data['firstzero'] = data['firstzero'].to(device)
                    x_recon, z = model(data['firstzero'])
                    last_coordinate = z[0, :]
                    loss_zero = torch.nn.functional.mse_loss(10*last_coordinate, torch.zeros_like(last_coordinate))
                    losses['firstzero'] = loss_zero

                if isEnabled('polars'):
                    data['polars'] = data['polars'].to(device)
                    x_recon, z = model(data['polars'])
                    last_coordinate = z[-1, :]
                    first_coordinate = z[0, :]
                    loss_zero = torch.nn.functional.mse_loss(10*last_coordinate, 10*0.8*torch.ones_like(last_coordinate))
                    loss_zero2 = torch.nn.functional.mse_loss(10*first_coordinate, 10*-0.8*torch.ones_like(first_coordinate))
                    losses['polars'] = loss_zero + loss_zero2

                if isEnabled('wellspacedtrajectory'):
                    x_recon, z = model(data['wellspacedtrajectory'].to(device))
                    losses['wellspacedtrajectory'] = loss_well_spaced_trajectory(z)
                
                if isEnabled('gridscaling'):
                    data_grid_latent, data_trajectory = data['gridscaling']
                    data_grid_latent = data_grid_latent[0] # because of TesnorDataset
                    data_grid_latent = data_grid_latent.to(device)
                    data_trajectory = data_trajectory.to(device)
                    losses['gridscaling'] = loss_grid_to_trajectory(model, data_grid_latent, data_trajectory, d_max_inputspace, args.grid_mode, args.grid_ratio, epoch=epoch, latentfactor=args.latentfactor)
                
                loss_total_batch = 0
                for i in losses:
                    weighted_loss = losses[i]*loss_dict[i]['weight']
                    loss_total_batch += weighted_loss
                    total_losses[i] += weighted_loss.item()                   
                total_loss += loss_total_batch.item()


                loss_total_batch.backward()
                optimizer.step()
                scheduler.step(epoch + batch_idx/max_batches)
        else:
            break

        
        for i in losses:
            total_losses[i] = total_losses[i]/max_batches     
        total_loss = total_loss/max_batches


        row = {'epoch': epoch}
        for i in loss_dict.keys():
            row[loss_dict[i]['official_name']] = total_losses[i]
        row['Total loss'] = total_loss
        row['Learning rate'] = scheduler.get_last_lr()[0] 
        df_losses = df_losses.append(row, ignore_index=True)
    

        if epoch%args.every_epoch == 0:
            earlystopping.on_epoch_end(epoch, total_loss, model)

            print('normalzied model data')
            df_errors = print_errors(rec_data_loader, model, device, best_model_path_directory, df=df_errors, epoch=epoch, err_tolerance=0.01, unnormalize=False)
            print('unnormalzied model data')
            df_errors_unnormalized = print_errors(rec_data_loader, model, device, best_model_path_directory, df=df_errors_unnormalized, epoch=epoch, err_tolerance=1, unnormalize=True)
            print('------')

            printed_string = f"Epoch: {epoch}\t"
            for i in loss_dict:
                if loss_dict[i]['weight']>0:
                    printed_string += f"{loss_dict[i]['official_name']}: {total_losses[i]:.4f}\t"
            printed_string += f"Total: {total_loss:.4f}"

            print(printed_string)

            df_losses.to_csv(os.path.join(best_model_path_directory, 'losses.csv'), index=False)
            filtered_columns = ['epoch', 'Total loss']
            for i in losses.keys():
                filtered_columns = filtered_columns + [loss_dict[i]['official_name']]
            plot_losses(df_losses[filtered_columns], args.every_epoch, best_model_path_directory)

    best_model = earlystopping.on_train_end()

    if isEnabled('anchor'):
        print_coordinates(loss_dict['anchor']['dataloader'], best_model, predefined_values, device, best_model_path_directory)

