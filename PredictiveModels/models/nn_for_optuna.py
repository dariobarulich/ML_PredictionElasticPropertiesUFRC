#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
# ======================================================================== INI79
##
# defining NN for Optuna-.
#
# @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
# start date (Arg): lun 02 sep 2024 14:32:33 -03-.
# last modify (Arg): vie 28 feb 2025 16:00:58 -03-.
##
# ======================================================================== END79


# ======================================================================== INI79
# include packages/modules/variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# 2don't have problems between float datatype of torch module and bnnNN module
# if torch.is_tensor(xx) else torch.tensor(xx,dtype=float) (float64) and float
# of NN (float32)-.
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
# - ====================================================================== END79

# define the Neural Network (NN) model-.
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units,
                 dropout_prob, learning_rate, optimizer, loss, weight_decay,
                 momentum):
        
        super(RegressionModel, self).__init__()
        self.layers = []
        in_features = input_size
        
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(in_features, hidden_units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))
            in_features = hidden_units

        self.layers.append(nn.Linear(in_features, output_size))  # output layer-.

        self.network = nn.Sequential(*self.layers)

        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=self.learning_rate)
        self.loss = loss  # nn.MSELoss()
        self.wd = weight_decay
        self.momentum = momentum
        
        
    def forward(self, x):
        return self.network(x)
    
    
    def save_params_model(self, dir_logs, bst, bsv, epochs, epochs_optuna,
                          optimizer, loss, ds_file, dir_res, case) -> None:
        ''' method to save the DL model's main params '''
        
        # print a loogBookFile-.
        # print(dir_logs)
        log_file = dir_logs+'/' + 'nn' + case + '_log.p'; file_log = open(log_file, 'wb')
        # print(log_file, type(log_file), sep='\n')
        # '/gpfswork/rech/irr/uhe87yl/carazof/scratch/fernando/resDL_2_SS/log.p'

        # dictionary2print-.
        log_dict = {'learning_rate': self.learning_rate,
                    'batch_size_train': bst,
                    'batch_size_val': bsv,
                    'num_epochs_optuna': epochs_optuna,
                    'num_epochs': epochs,
                    'layers_list': self.layers,
                    'optimizer': optimizer,  # (str) I don't save self.optimizer 2don't save torch.optim ..
                    'weight_decay': self.wd,
                    'momentum': self.momentum,
                    'loss': loss,  # (str) I don't save self.loss 2don't save torch.optim ..
                    'dataset_file': ds_file,
                    'folder_out': dir_res,
                    }
        
        # with open(log_file,'wb') as fn: pickle.dump(log_dict,file_log); fn.close()
        pickle.dump(log_dict, file_log)
        file_log.close()

        return None

if __name__ == '__main__':
    pass
