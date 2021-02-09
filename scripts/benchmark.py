import sys, ast, os
import time
import pickle
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
from plotnine import *
import matplotlib.pyplot as plt 
import matplotlib
data_type = 'float32'
os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=' + data_type + ',force_device=True' + ',dnn.enabled=False'
# /nfs/team283/vk7/software/miniconda3farm5/envs/cellpymc/bin/pip install git+https://github.com/vitkl/cell2location.git
path = '/nfs/team283/aa16/'
os.chdir('/nfs/team283/aa16/KR_NAS/')
from matplotlib import rcParams
import seaborn as sns
# scanpy prints a lot of warnings
import warnings
warnings.filterwarnings('ignore')
sys.path.append('/nfs/team283/aa16/KR_NAS')
from nanostringWTA import WTACounts_GeneralModel

x = int(sys.argv[1])

data = ('only celltype AOIs', 'only celltype AOIs', 'only celltype AOIs', 'only celltype AOIs',
        'all 19pcw technical replicates', 'all 19pcw technical replicates', 'all 19pcw technical replicates', 'all 19pcw technical replicates',
        'all data', 'all data', 'all data', 'all data')
factors = (5,10,20,30,5,10,20,30,5,10,20,30)

d = data[x]
f = factors[x]

adata = pickle.load(open(path + "KR_NAS/data/nanostringWTA_fetailBrain_AnnData.p", "rb" ))

if d == 'only celltype AOIs':
    age = '19pcw'
    radial_position = (1,3)
    aoi_type = ('EOMESpos', 'HOPXpos')
    subset = np.array([adata.obs['Radial_position'][i] in radial_position and
                       adata.obs['age'][i] == age and
                       adata.obs['AOI_type'][i] in aoi_type for i in range(len(adata.obs['age']))])
    adata = adata[subset,:]
elif d == 'all 19pcw technical replicates':
    age = '19pcw'
    radial_position = (1,2,3)
    subset = np.array([adata.obs['Radial_position'][i] in radial_position and
                       adata.obs['age'][i] == age for i in range(len(adata.obs['age']))])
    adata = adata[subset,:]

X_data = np.asarray(adata[:,np.array(adata.var != 'NegProbe-WTX').squeeze()].X)
Y_data = np.asarray(adata[:,np.array(adata.var == 'NegProbe-WTX').squeeze()].X)

mod1 = WTACounts_GeneralModel(
        X_data, Y_data,
        data_type='float32', n_iter=1,
        learning_rate=0.001,
        total_grad_norm_constraint=200,
        n_factors = f,
        verbose=False, var_names= np.array(adata.var).squeeze()[np.array(adata.var).squeeze() != 'NegProbe-WTX'],
                                                                obs_names=np.array(adata.obs.index))

def get_wta_results():
    import theano
    theano.config.compute_test_value = 'ignore'
    mod1.sample_posterior(node='all', n_samples=1000, save_samples=False);
    A = [adata.obs['AOI_type'][i] == 'EOMESpos' and adata.obs['slide'][i] in ('00MU') for i in range(len(adata.obs['AOI_type']))]
    B = [adata.obs['AOI_type'][i] == 'HOPXpos' and adata.obs['slide'][i] in ('00MU') for i in range(len(adata.obs['AOI_type']))]
    mod1.compute_logFC(groupA = np.squeeze(np.where(A)),groupB = np.squeeze(np.where(B)),n_samples = 1000)
    mod1.compute_FDR(logFC_threshold = 1)
    return(mod1.logFC)

mod1.fit_advi_iterative(n=1, method='advi', n_iter = 1000)
wta_results = get_wta_results()
pickle.dump(wta_results, open("/nfs/team283/aa16/KR_NAS/savedModels/benchmarking/WTACounts_GeneralModel_Results_parameterSet" + str(x) + "n_iter1000.p", "wb" ))
mod1.fit_advi_refine(n_iter = 4000)
wta_results = get_wta_results()
pickle.dump(wta_results, open("/nfs/team283/aa16/KR_NAS/savedModels/benchmarking/WTACounts_GeneralModel_Results_parameterSet" + str(x) + "n_iter5000.p", "wb" ))
mod1.fit_advi_refine(n_iter = 5000)
wta_results = get_wta_results()
pickle.dump(wta_results, open("/nfs/team283/aa16/KR_NAS/savedModels/benchmarking/WTACounts_GeneralModel_Results_parameterSet" + str(x) + "n_iter10000.p", "wb" ))
for i in range(9):
    mod1.fit_advi_refine(n_iter = 10000)
    wta_results = get_wta_results()
    pickle.dump(wta_results, open("/nfs/team283/aa16/KR_NAS/savedModels/benchmarking/WTACounts_GeneralModel_Results_parameterSet" + str(x) + "n_iter" + str(10000 + (i+1)*10000) +".p", "wb" ))
for i in range(6):
    mod1.fit_advi_refine(n_iter = 25000)
    wta_results = get_wta_results()
    pickle.dump(wta_results, open("/nfs/team283/aa16/KR_NAS/savedModels/benchmarking/WTACounts_GeneralModel_Results_parameterSet" + str(x) + "n_iter" + str(100000 + (i+1)*25000) +".p", "wb" ))
for i in range(6):
    mod1.fit_advi_refine(n_iter = 50000)
    wta_results = get_wta_results()
    pickle.dump(wta_results, open("/nfs/team283/aa16/KR_NAS/savedModels/benchmarking/WTACounts_GeneralModel_Results_parameterSet" + str(x) + "n_iter" + str(250000 + (i+1)*50000) +".p", "wb" ))
for i in range(5):
    mod1.fit_advi_refine(n_iter = 100000)
    wta_results = get_wta_results()
    pickle.dump(wta_results, open("/nfs/team283/aa16/KR_NAS/savedModels/benchmarking/WTACounts_GeneralModel_Results_parameterSet" + str(x) + "n_iter" + str(500000 + (i+1)*100000) +".p", "wb" ))
    pickle.dump(mod1, open("/nfs/team283/aa16/KR_NAS/savedModels/benchmarking/WTACounts_GeneralModel_FullModel_parameterSet" + str(x) + "n_iter" + str(500000 + (i+1)*100000) +".p", "wb" ))
