import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from PredictiveModels.ElasticModuli import romBased as ROM
# from PredictiveModels.ElasticModuli import chamisModel as chM
# from PredictiveModels.baseModel_ANN import ANNmodel as ann

from models.ElasticModuli import romBased as ROM
from models.ElasticModuli import chamisModel as chM
from models.baseModel_ANN import ANNmodel as ann

from datetime import *

MARKERList = ['h','^','v','s','d','.','<','o','<',]

listOfColors = ['b','g','r','y','m','k','c','m','b',]

def getTime():
    d=datetime.today()
    anio=d.year
    dia=d.day
    mes=d.month
    hora=d.hour
    minutos=d.minute
    segundos=d.second
    fechaN = str(dia)+'m'+str(mes)+'_'+str(hora)+'m'+str(minutos)+'s'+str(segundos)
    return fechaN

def getabsolErr(y_target, y):
    absolErr = np.average(np.abs(y_target-y)/y_target*100.0)
    return absolErr

def getMRE(y_target, y):
    MRE = np.average((y_target-y)/y_target*100.0)
    return MRE

def getRMSE(y_target, y):
    RMSE = np.sqrt(np.sum((y_target-y)**2)/len(y_target))
    return RMSE

def getMSE(y_target, y):
    MSE = np.sum((y_target-y)**2)/len(y_target)
    return MSE

def getCumHist(y_target, y, ax, modName,color_i,marker_i,errorName):
    errorName = 'MRE'
    if errorName == 'MSE':
        RMSE = getMSE(y_target, y)
    if errorName == 'RMSE':
        RMSE = getRMSE(y_target, y)
    if errorName == 'MRE':
        RMSE = np.abs(getMRE(y_target, y))
        x = (y-y_target)/y_target*100.0
    print(f'  {modName}  {errorName}= {RMSE:8.4e}')

    stats = (rf'({RMSE:8.4e})')
    
    legName = modName+stats
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    n,bins,patches = ax2.hist(x, bins=60, density=False, cumulative=False, histtype="step")
    plt.close(fig2)

    n_points = len(n)

    n_markers = 25  
    step = n_points // (n_markers)  # Calculate step size for equal spacing
    marker_indices = np.arange(0, n_points, step)[:n_markers]  # Select indices at intervals

    bisL = bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
    ax.plot(bisL, n, markerfacecolor= 'white', color= color_i, marker= marker_i,
            label=legName, markevery = int(n_points/25)) # color_i,marker_i


if __name__ == '__main__':

    # Load data from CSV
    print('Opening DataBase')
    data = pd.read_csv('./data/database_short.csv')  # Assume CSV has columns x1, x2, y
    
    # Extracting features and target
    EM, nuM, EF, nuF, Vf = data['EM'].values, data['nuM'].values, data['EF'].values, data['nuF'].values, data['Vf'].values

    #=====================================================
    # EM,nuM,EF,nuF,Vf
    E1f = np.array(data['EF'])
    E2f = np.array(data['EF'])
    nu12f = np.array(data['nuF'])
    nu23f = np.array(data['nuF'])
    G12f = E1f/(2.0*(1.0+nu12f))
    Em = np.array(data['EM'])
    num = np.array(data['nuM'])
    Vf = np.array(data['Vf'])

    numModel = 0
    lisMod = []
    modelNames = []

    modName = 'Ch    '
    numModel = numModel +1
    chm_df = chM(E1f,E2f,G12f,nu12f,nu23f,Em,num,Vf)
    lisMod.append(chm_df)
    modelNames.append(modName)
    print(modName)
    # getCumHist(y_target, chm_df[propName], ax, modName)
    
    modName = 'ROM   '    
    numModel = numModel +1
    rom_df = ROM(E1f,E2f,G12f,nu12f,nu23f,Em,num,Vf)
    lisMod.append(rom_df)
    modelNames.append(modName)
    print(modName)
    # getCumHist(y_target, rom_df[propName], ax, modName)
    
    modName = 'BaseL '
    numModel = numModel +1
    print(modName)
    ann_df = ann(Em,num,E1f,nu12f,Vf)
    lisMod.append(ann_df)
    modelNames.append(modName)
    input(11)
    
    PropNames = ['E1', 'E2', 'v12', 'G12', 'G23',]
    PropNamesM = ['E1', 'E2', 'nu12', 'G12', 'G23',]
    labPropNamesM = ['E_{11', 'E_{22', '\\nu_{12', 'G_{12', 'G_{23',]
    scalefact = [  1.0,  1.0,    1.0, 1.0/2.0, 1.0/2.0]
    
    errorName = 'MRE'
    
    mapNames = ['E1','E2','nu12','G12','G23',]

    plt.rcParams.update({
                        "text.usetex": True,
                        "font.family": "serif",
                        "font.serif": ["Computer Modern Roman"],
                        "axes.labelsize": 14,
                        "axes.titlesize": 16,
                        "legend.fontsize": 12,
                        "xtick.labelsize": 12,
                        "ytick.labelsize": 12,
                    })
    fig = plt.figure(figsize=(12, 12), layout='constrained')
    axs = fig.subplot_mosaic([
                            ["E2", "nu12"],
                            ["G12", "G23"],
                            ["E1", "E1"]])

    for k, propName in enumerate(PropNames):
        
        ax = axs[mapNames[k]]
        # Set bottom and left spines to black
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        propNameM = PropNamesM[k]
        labPropName = labPropNamesM[k]
        
        # ==========================================================
        # get order following errorName
        lisRMSE = []
        llaveS = '}'
        y_targ = np.array(data[propName].values*scalefact[k])
        print('===============================================')
        print(f'Property: {propName}')
        for j, modelName in enumerate(modelNames):
            y = np.array(lisMod[j][propNameM])
            # RMSE = getRMSE(dfL[i][colExp], lisMod[j][colExp])
            if errorName == 'MSE':
                RMSE = getMSE(y_targ, y)
            if errorName == 'RMSE':
                RMSE = getRMSE(y_targ, y)
            if errorName == 'MRE':
            # RMSE = getRMSE(y_targ, y)
                # RMSE = getabsolErr(y_targ, y)
                RMSE = np.abs(getMRE(y_targ, y))
            
            lisRMSE.append(RMSE)
        # Get the indices that would sort the list
        original_indices = np.argsort(lisRMSE)
        # ==========================================================
        for j in original_indices:
            color_i,marker_i = listOfColors[j], MARKERList[j]
            getCumHist(data[propName].values*scalefact[k], lisMod[j][propNameM], ax, modelNames[j],color_i,marker_i,errorName)
            ax.grid(True)
            ax.set_ylabel("\\textbf{frequency}")

        titleFig = "Model (\\textbar MRE \\textbar)"
        legend_font = {'family': 'Courier New', 'size': 12}
        legend = ax.legend(title = titleFig, prop = legend_font)
        ax.set_xlabel(rf'$\frac{{{labPropName}{llaveS} - {labPropName}(CM){llaveS}}}{{{labPropName}(CM){llaveS}}} 100$ [\%]')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_facecolor('white')
        

    fig.savefig('./figs/'+'CompProps_'+'MRE'+'_'+getTime()+'.pdf') 

plt.show()
