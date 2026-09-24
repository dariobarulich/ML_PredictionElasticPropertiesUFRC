"""
 Provides Models for predicting elastic moduli of unidirectional fiber reinforced composites.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scipy.linalg import solve




def romBased(
        E1f: NDArray, 
        E2f: NDArray, 
        G12f: NDArray,
        nu12f: NDArray,
        nu23f: NDArray,
        Em: NDArray,
        num: NDArray,
        Vf: NDArray
        ) -> pd.DataFrame:
    """ Computes the macro elastic moduli using the Rule of Mixtures (and Inverse Rule of Mixtures).
    Equations taken from Vignoli, L. L., Savi, M. A., Pacheco, P. M., & Kalamkarov, A. L. (2019).
    Comparative analysis of micromechanical models for the elastic composite laminae. Composites Part B: Engineering, 174, 106961

    Args:
        E1f (NDArray): Young's modulus of the fiber in the longitudinal direction.
        E2f (NDArray): Young's modulus of the fiber in the transverse direction.
        G12f (NDArray): Shear modulus of the fiber in the axial-transverse direction.
        nu12f (NDArray): Poisson's ratio of the fiber.
        nu23f (NDArray): Poisson's ratio of the fiber.
        Em (NDArray): Young's modulus of the matrix.
        num (NDArray): Poisson's ratio of the matrix.
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
    # Ensure valid volume fractions
    if np.any((Vf < 0) | (Vf > 1)):
        raise ValueError("Fiber volume fraction (Vf) must be between 0 and 1.")

    # Shear modulus of the matrix
    Gm = Em/(2.0*(1.0+num))

    # bulk modulus of the matrix
    Km = Em/(3.0*(1.0-2.0*num))
    

    # transverse bulk modulus of the fiber (TI) VERIFY THIS EXPRESSION
    K23f = E2f/(3.0*(1.0-2.0*nu23f))

    K23_e = K23f*Km/(K23f*(1.0-Vf) + Km*Vf)

    # Moduli
    E1_e = E1f*Vf + (1.0-Vf)*Em
    E2_e = E2f*Em/(E2f*(1.0-Vf) + Em*Vf)
    E3_e = E2_e
    G12_e = G12f*Gm/(G12f*(1.0-Vf) + Gm*Vf)
    G13_e = G12_e
    
    nu12_e = nu12f*Vf + (1.0 - Vf)*num
    G23_e = E1_e/(4.0*(E1_e/E2_e - nu12_e**2.0) - E1_e/K23_e)
    nu13_e = nu12_e
    nu23_e = E2_e/G23_e/2.0 -1.0


    # Create a DataFrame
    df = pd.DataFrame({
                'E1f': E1f,
                'E2f': E2f,
                'G12f': G12f,
                'nu12f': nu12f,
                'nu23f': nu12f,
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

def chamisModel(
        E1f: NDArray, 
        E2f: NDArray, 
        G12f: NDArray,
        nu12f: NDArray,
        nu23f: NDArray,
        Em: NDArray,
        num: NDArray,
        Vf: NDArray
    ) -> pd.DataFrame:
    
    """
    Computes the macro elastic moduli using the Chamis Model (microstructural voids are neglected, V_v = 0).
    Equations taken from Vignoli, L. L., Savi, M. A., Pacheco, P. M., & Kalamkarov, A. L. (2019).
    Comparative analysis of micromechanical models for the elastic composite laminae. Composites Part B: Engineering, 174, 106961

    Args:
        E1f (NDArray): Young's modulus of the fiber in the longitudinal direction.
        E2f (NDArray): Young's modulus of the fiber in the transverse direction.
        G12f (NDArray): Shear modulus of the fiber in the axial-transverse direction.
        nu12f (NDArray): Poisson's ratio of the fiber.
        nu23f (NDArray): Poisson's ratio of the fiber.
        Em (NDArray): Young's modulus of the matrix.
        num (NDArray): Poisson's ratio of the matrix.
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
    # Ensure valid volume fractions
    if np.any((Vf < 0) | (Vf > 1)):
        raise ValueError("Fiber volume fraction (Vf) must be between 0 and 1.")

    # Shear modulus of the matrix
    Gm = Em/(2.0*(1.0+num))

    G23f = E2f/(2.0*(1.0+nu23f))
    
    
    # Moduli
    E1_e = E1f*Vf + (1.0-Vf)*Em
    E2_e = Em/(1.0-np.sqrt(Vf)*(1.0-Em/E2f))
    E3_e = E2_e
    G12_e = Gm/(1.0-np.sqrt(Vf)*(1.0-Gm/G12f))
    G13_e = G12_e
    
    nu12_e = num+Vf*(nu12f-num)
    G23_e = Gm/(1.0-np.sqrt(Vf)*(1.0-Gm/G23f))
    nu13_e = nu12_e
    nu23_e = E2_e/G23_e/2.0 -1.0

    


    # Create a DataFrame
    df = pd.DataFrame({
                'E1f': E1f,
                'E2f': E2f,
                'G12f': G12f,
                'nu12f': nu12f,
                'nu23f': nu12f,
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
