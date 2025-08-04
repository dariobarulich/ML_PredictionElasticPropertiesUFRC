"""
Composite Models: provides tools for predicting mechanical behavior of composite materials.
"""

from .ElasticModuli import romBased, chamisModel


# from .baseModel_ANN import ANNmodel

from .baseModel_ANN import ANNmodel




# Documentation of submodules
__doc__ += f"""

Submodules:
-----------
- ElasticModuli: {ElasticModuli.__doc__}
- baseModel_ANN: {baseModel_ANN.__doc__}

"""

