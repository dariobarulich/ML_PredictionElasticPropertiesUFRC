"""
 The base model using ANN developed by Dr. Carazo. 14 May 2025
"""

import torch
import pickle
import numpy as np
# from models.ffnn import MLP
# from ffnn import MLP
from PredictiveModels.models.ffnn import MLP
# from models.ffnn import MLP
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray


import warnings
warnings.filterwarnings('ignore')

# Load a scaler from a .pkl file
def load_scaler(pkl_path):
    with open(pkl_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def load_model(model_path):
    input_size, output_size = 5, 8
    layers = [(1024, nn.ReLU(), None),  # for now it isn't work from yaml file-.
            # (256, nn.ReLU(), None),  # for now it isn't work from yaml file-.
            # (1024,nn.ReLU(),None),  # for now it isn't work from yaml file-.
            (output_size, None, None)]
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    # print(f'dev: {device}')
    model = MLP(input_size,
                output_size,
                layers,
                device,
                1.0e-3,
                torch.optim.Adam,
                nn.MSELoss(),
                0.0,
                0.99)
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to preprocess input data using the feature scaler
def preprocess_data(data, feature_scaler):
    data_scaled = feature_scaler.transform(data)
    return torch.FloatTensor(data_scaled)

# Function to make predictions and inverse scale if needed
def make_predictions(model, input_data, target_scaler=None):
    input_data = torch.tensor(input_data, dtype=torch.float64)
    with torch.no_grad():  # No gradient computation for inference
        predictions = model(input_data)
        if target_scaler:
            predictions = target_scaler.inverse_transform(predictions.numpy())
        else:
            predictions = predictions.numpy()
    return predictions



def ANNmodel(        
        Em: NDArray,
        num: NDArray,
        Ef: NDArray, 
        nuf: NDArray,
        Vf: NDArray
        # ) -> List[NDArray]:
        ) -> pd.DataFrame:
    """ Computes the macro elastic moduli of a unidirectional composite material using an ANN model build by Dr. F. Carazo (14 May 25)

    Args:
        Em (NDArray): Young's modulus of the matrix.
        num (NDArray): Poisson's ratio of the matrix.
        Ef (NDArray): Young's modulus of the fiber.
        nuf (NDArray): Poisson's ratio of the fiber.
        Vf (NDArray): Fiber volume fraction .

    Returns:
        DataFrame[NDArray, ]:
            - 'E1': Macro Longitudinal Young's modulus.
            - 'E2': Macro Tranverse Young's modulus.
            - 'E3': Macro Tranverse Young's modulus.
            - 'G12': Macro Shear modulus in the axial-transverse direction.
            - 'G13': Macro Shear modulus in the axial-transverse direction.
            - 'G23': Macro Shear modulus in the transverse direction.
            - 'nu12': Macro Poisson's ratio of the fiber.
            - 'nu13': Macro Poisson's ratio of the fiber.
            - 'nu23': Macro Poisson's ratio of the fiber.
    """

    print('ANN')

    # pathTofiles = 'N:/2024/AI_composites/Codes/PaperVersion/PredictiveModels/baselineModel/'
    # pathTofiles = '/home/fdcarazo/my_github/compElasProp/'
    pathTofiles = './models/'
    feature_scaler_path = pathTofiles+'featScaler.pkl'
    target_scaler_path = pathTofiles+'targScaler.pkl'   
    model_path = pathTofiles+'dlModelWithoutHyperOpt_sd.pt'

    # Load scalers and model
    feature_scaler = load_scaler(feature_scaler_path)
    target_scaler = load_scaler(target_scaler_path)
    model = load_model(model_path)
    
    input_data = np.column_stack((Em, num, Ef, nuf, Vf))

    # Preprocess the input data
    scaled_input = preprocess_data(input_data, feature_scaler)
    
    # Make predictions
    # outputs: ['E1', 'E2', 'v12', 'v23', 'G12',  'G23', 'E3', 'G13']
    predictions = make_predictions(model, scaled_input, target_scaler)
    
    Gm = Em/2.0/(1.0+num)
    Gf = Ef/2.0/(1.0+nuf)
    E1_e = predictions[:,0]
    E2_e = predictions[:,1]
    E3_e = E2_e
    G12_e = predictions[:,4]/2.0
    G13_e = G12_e
    G23_e = predictions[:,5]/2.0
    nu12_e = predictions[:,2]
    nu13_e = nu12_e
    nu23_e = predictions[:,3]

    # Create a DataFrame
    df = pd.DataFrame({
                'E1f': Ef,
                'E2f': Ef,
                'G12f': Gf,
                'nu12f': nuf,
                'nu23f': nuf,
                'Em': Em,
                'num': num,
                'Vf': Vf,
                #
                'E1': E1_e,
                'E2': E2_e,
                'E3': E3_e,
                'G12': G12_e,
                'G13': G13_e,
                'G23': G23_e,
                'nu12': nu12_e,
                'nu13': nu13_e,
                'nu23': nu23_e
                })


    return df

if __name__ == "__main__":
    
    pass
