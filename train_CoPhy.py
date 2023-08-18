import argparse
import json
import pandas as pd
import torch
import os
import itertools
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import warnings
warnings.filterwarnings("ignore", message="The frame.append method is deprecated")


from aux.AEmodel import UniformAutoencoder
from aux.earlystopping import EarlyStopping
from aux.losses import loss_grid_to_trajectory, rec_loss_function, loss_anchor
from aux.trajectories_data import get_trajectory_dataloader, get_anchor_dataloader, get_predefined_values
from aux.utils import get_files, get_gridpoint_and_trajectory_datasets, plot_losses



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoPhy training')

    #AE archi
    parser.add_argument('--num_of_layers', type=int, default=3)
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
    parser.add_argument('--anchor_mode', default="diagonal", type=str, help='anchor shape') #"circle"#"diagonal"
    parser.add_argument('--d_max_latent', default=2, type=float, help='d_max_latent') #"proportional"#"scaled"

    #weigths
    parser.add_argument('--rec_weight', default=1.0, type=float)
    parser.add_argument('--anchor_weight', default=0.0, type=float)
    parser.add_argument('--lastzero_weight', default=0.0, type=float)
    parser.add_argument('--gridscaling_weight', default=0.0, type=float)

    latent_dim = 2

    args = parser.parse_args()

    file_path = args.model_folder
    best_model_path = args.model_file

    ###### HYPERP ######
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_dict = {
        'rec': {
            'weight': args.rec_weight,
            'official_name': "Reconstruction loss"
        },
        'anchor': {
            'weight': args.anchor_weight,
            'official_name': "Anchor loss"
        },
        'lastzero': {
            'weight': args.lastzero_weight,
            'official_name': "LastZero loss"
        },
        'gridscaling': {
            'weight': args.gridscaling_weight,
            'official_name': "Grid-scaling loss"
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


    if isEnabled('anchor'):
        anchor_dataloader = get_anchor_dataloader(dataset, range_of_files_for_anchor)
        loss_dict['anchor']['dataloader'] = anchor_dataloader
        predefined_values = get_predefined_values(anchor_dataloader.dataset, args.anchor_mode)
        predefined_values = predefined_values.to(device)

    if isEnabled('lastzero'):
        loss_dict['lastzero']['dataloader'] =  get_anchor_dataloader(dataset)
    
    l_max_inputspace=None
    if isEnabled('gridscaling'):
        loss_dict['gridscaling']['dataloader'] = get_gridpoint_and_trajectory_datasets(pt_files, best_model_path_directory, args.grid_step, batch_size=args.batch_size)
        data_trajectory_dataset_temp = loss_dict['gridscaling']['dataloader'].dataset
        data_trajectory_dataset_temp_0 = data_trajectory_dataset_temp[0][1]
        data_trajectory_dataset_temp_last = data_trajectory_dataset_temp[-1][1]
        l_max_inputspace = torch.sqrt((data_trajectory_dataset_temp_0 - data_trajectory_dataset_temp_last).pow(2).sum(dim=-1)).to(device)

    model = UniformAutoencoder(input_dim, args.num_of_layers, latent_dim).to(device)
    print('model', model)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, args.cosine_Scheduler_patience)



    #### Training

    def cycle_dataloader(dataloader):
        """Returns an infinite iterator for a dataloader."""
        return itertools.cycle(iter(dataloader))


    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        best_model = model
    else:
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

        
        columns.append('total')
        df_losses = pd.DataFrame(columns=columns)
        
        for epoch in range(args.epochs):
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
                        loss_zero = torch.nn.functional.mse_loss(last_coordinate, torch.zeros_like(last_coordinate))
                        losses['lastzero'] = loss_zero

                    if isEnabled('gridscaling'):
                        data_grid_latent, data_trajectory = data['gridscaling']
                        data_grid_latent = data_grid_latent[0] # because of TesnorDataset
                        data_grid_latent = data_grid_latent.to(device)
                        data_trajectory = data_trajectory.to(device)

                        losses['gridscaling'] = loss_grid_to_trajectory(model, data_grid_latent, data_trajectory, l_max_inputspace, epoch=epoch, d_max_latent=args.d_max_latent)
                    
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
            df_losses = df_losses.append(row, ignore_index=True)


            if epoch%args.every_epoch == 0:
                earlystopping.on_epoch_end(epoch, total_loss, model)

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

