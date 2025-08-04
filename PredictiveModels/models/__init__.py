"""
Defines Artificial neural network models.
"""
from .ffnn import *
from .ffnn_bnn import *
from .nn_for_optuna import *


# Documentation of submodules
__doc__ += f"""

Submodules:
-----------
- ANN: {ffnn.__doc__}
- BNN: {ffnn_bnn.__doc__}
- with hyperparameter optimization: {nn_for_optuna.__doc__}

"""
