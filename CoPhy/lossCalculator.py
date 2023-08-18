# Code is copied with necessary refactoring from https://github.com/jayroxis/Cophy-PGNN 

from CoPhy.model_params import Loss as modelparamsLoss, wrap_model
import torch
import math
from CoPhy.presets import getPresets
from CoPhy.data_loader import DatasetLoader

def inverse_norm(batch, scale, mean):
    return batch * scale + mean

def energy_loss(batchPred, batchInput):
    batchEg = batchPred[:, -1]
    loss_e = torch.exp(batchEg)
    return loss_e

def phy_loss(batchPred, batchReal, batchInput, norm=False):
    if batchInput.dim() == 2:
        H_height = int(math.sqrt(batchInput.size(1)))
        H_width = H_height
    elif batchInput.dim() == 3:
        H_height = batchInput.size(1)
        H_width = batchInput.size(2)
        
    batchC = batchPred[:, 0: H_width]
    batchEg = batchPred[:, -1]
    loss_phy = torch.sum(
            (
                torch.sum(
                    batchInput.view((-1, H_width, H_height)) * batchC.view((-1, 1, H_width)), 
                    dim=2
                ) - batchC * batchEg.view((-1, 1))
            ) ** 2, 
            dim = 1
        )
    if norm:
        loss_phy /= torch.sum(batchC ** 2, dim=1)
    return loss_phy

def loss_func(data, loss_list, outputs, e_coff=0.0, s_coff=1.0, batchX=None, batchY=None, batchH=None, norm=False):
    """ 
    Set batchY to None when train on test set. 
    Set batchX to None when only use MSE.
    """
    
    # MSE Loss
    criterion = torch.nn.MSELoss()
    if (batchY is not None) and ('mse_loss' in loss_list):
        loss = criterion(outputs, batchY)
    else:
        loss = 0.0

    if batchH is not None:
        origin_input = batchH
    else:
        # inverse transformation for normalization
        if data.std_scaler_x is not None:
            origin_input = inverse_norm(
                batchX, data.X_scale_tensor, data.X_mean_tensor
            )
        else:
            origin_input = batchX

    if data.std_scaler_y is not None:
        origin_output = inverse_norm(
            outputs, data.y_scale_tensor, data.y_mean_tensor
        )
        origin_y = inverse_norm(
            batchY, data.y_scale_tensor, data.y_mean_tensor
        )
    else:
        origin_output = outputs
        origin_y = batchY

    # physics loss and energy loss
    if 'phy_loss' in loss_list:
        loss_phy = phy_loss(
            origin_output,
            origin_y,
            origin_input,
            norm=norm
        )
    else:
        loss_phy = 0.0

    if 'energy_loss' in loss_list:
        loss_e = energy_loss(
            origin_output,
            origin_input
        )
    else:
        loss_e = 0.0
        
    if type(loss_phy) == torch.Tensor or type(loss_e) == torch.Tensor:
        loss += torch.mean(s_coff * loss_phy + e_coff * loss_e)

    norm_loss_phy = phy_loss(
        origin_output,
        origin_y,
        origin_input,
        norm=True
    )
    norm_loss_phy = torch.mean(norm_loss_phy)
    loss_phy = phy_loss(
        origin_output,
        origin_y,
        origin_input,
        norm=False
    )
    loss_phy = torch.mean(loss_phy)
    loss_e = energy_loss(
        origin_output,
        origin_input
    )
    loss_e = torch.mean(loss_e)
    return loss, loss_phy, norm_loss_phy, loss_e


norms = {}
DNN_types=["NN", "PGNN_OnlyDTr", "PGNN_LF", "PGNN_"]
for DNN_type in DNN_types:
    (train_loss_list, test_loss_list, norm) = getPresets(DNN_type)
    norms[DNN_type] = norm

class PhyLoss(modelparamsLoss):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, dnnType, model, inputs: torch.Tensor, target: torch.Tensor, X_trainOrigin, X_test, y_test, X_testOrigin, datasetLoader, e_coff=0, s_coff=0.846):
        super().__init__(None, inputs, target)
        self.inputs = inputs
        self.target = target
        self.XtrainOrigin = X_trainOrigin
        self.dnnType = dnnType
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.X_testOrigin = X_testOrigin
        self.datasetLoader = datasetLoader

        self.e_coff = e_coff
        self.s_coff = s_coff

    def __call__(self, model_wrapper) -> float:
        self.inputs.grad = None
        
        self.predictions = model_wrapper.forward(self.inputs)

        loss_train, loss_phy_train, norm_loss_phy_train, loss_e_train = loss_func(self.datasetLoader, ['phy_loss'], self.predictions, 1, 1, self.inputs, self.target, self.XtrainOrigin, norms[self.dnnType])

        loss_test, loss_phy_test, norm_loss_phy_test, loss_e_test = loss_func(self.datasetLoader, ['energy_loss'], self.model(self.X_test), 1, 1, self.X_test, self.y_test, self.X_testOrigin, norms[self.dnnType])

        return {
            'phy': {
                'train_loss': loss_phy_train,
                'test_loss': loss_phy_test,
                'total': loss_phy_train+loss_phy_test
            },
            'e': {
                'train_loss': loss_e_train,
                'test_loss': loss_e_test,
                'total': loss_e_train+loss_e_test
            }
        }
    
class Loss:
    def __init__(self, DNN_type, dataPath, n_spins, trainingCount, validation_count, device):
        datasetLoader = DatasetLoader(dataPath, n_spins, trainingCount, validation_count, 0)
        datasetLoader.normalization(x=True, y=False)
        datasetLoader.torch_tensor(device=device)
        self.device = device

        self.DNN_type = DNN_type

        self.X_train = datasetLoader.X_train_tensor
        self.X_test = datasetLoader.X_test_tensor
        self.y_train = datasetLoader.y_train_tensor
        self.y_test = datasetLoader.y_test_tensor
        self.X_trainOrigin = datasetLoader.X_train_origin
        self.X_testOrigin = datasetLoader.X_test_origin

        self.datasetLoader = datasetLoader
        self.criterion = torch.nn.MSELoss()


    def get_loss(self, dnn, loss_name, whichloss):
        wrapped =  wrap_model(dnn)
        if whichloss=="phy" or whichloss=="e":
            metric_phy = PhyLoss(self.DNN_type,  dnn, self.X_train, self.y_train, self.X_trainOrigin, self.X_test, self.y_test, self.X_testOrigin, self.datasetLoader)
            return metric_phy(wrapped)[whichloss][loss_name]
        elif whichloss=="mse":
            if loss_name == "total":
                raise "Skipping total for mse loss"
            x = self.X_train if loss_name=="train_loss" else self.X_test
            y = self.y_train if loss_name=="train_loss" else self.y_test
            metric = modelparamsLoss(self.criterion, x, y)
            return metric(wrapped)
        else:
            raise "loss not defined"


    