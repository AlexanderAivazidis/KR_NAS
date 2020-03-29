## This scripts gets genes with spatial structures in the the 19pcw sample cell type specific ROIs and ring plus background using Vitalii's algorithm.
%pylab inline
import pandas as pd
import sys, ast, os
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
import pickle as pickle
import numpy as np
import time
import itertools
data_type = 'float32'
os.environ["THEANO_FLAGS"] = 'floatX=' + data_type #+ ',force_device=True'
%matplotlib inline
import anndata
import scanpy as sc
import theano.tensor as tt
import pymc3 as pm
#gfd
import theano
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import warnings
import os
import pycell2location.models as c2l
import pyreadr

os.chdir('/home/jovyan/KR_NAS/')

rSlides = np.array(('00MU', '00MV', '00MV-2')) # slide we want to look at
AOI_type = np.array(('EOMESpos', 'HOPXpos', 'Residual', 'Ring'))
Radial_positions = np.array((1,2,3))

# Let's load the data and metadata from our Nanostring experiment:
counts = pd.read_table('Sanger_288ROIs_TargetCountMatrix.txt')
genes = counts['TargetName']
counts = counts.drop('TargetName',1)
counts = counts.rename(index=genes)
counts = counts.astype('int')
counts = counts[np.array([sum(counts.iloc[i,:]) for i in range(len(counts.iloc[:,0]))]) > 3 * 288]
metadata = pd.read_csv('NanoString sequencing all annotations 2020.02.10.csv')
metadata = metadata.iloc[0:286,]
metadata = metadata.rename(index=metadata['Sanger_sampleID'])
metadata = metadata.reindex(np.array(counts.columns))
properties = pd.read_table('Sanger_288ROIs_SegmentProperties.txt')
properties = properties.rename(index=properties['DSP_Sample_ID'])
properties = properties.reindex(np.array(metadata['Sample_ID']))
columnNames = ('x', 'y', 'total_counts', 'Q3_counts')
sample_info = pd.DataFrame(index=metadata['Sample_ID'], columns=columnNames)
sample_info['x'] = np.array(metadata['VCDepth'])
sample_info['y'] = np.array(metadata['Radial_position'])
sample_info['total_counts'] = [sum(counts.iloc[:,i]) for i in range(len(counts.iloc[1,:]))] 
sample_info['Q3_counts'] = [sum(np.sort(counts.iloc[:,i])[int(np.round(0.5*len(counts.iloc[:,i]))):int(np.round(0.75*len(counts.iloc[:,i])))]) for i in range(len(counts.iloc[1,:]))] 

# Let's load the Polioudakis 2019 data as a reference, as well as the cell type specific marker genes:

polioudakis = pd.read_csv('/home/jovyan/data/fetalBrain/Polioudakis/cellStateMatrix.csv')
markers = pd.read_csv('/home/jovyan/data/fetalBrain/Polioudakis/clusterMarkers.csv')
genes = polioudakis.iloc[:,0]
polioudakis = polioudakis.drop('Unnamed: 0',1)
polioudakis = polioudakis.rename(index=genes)

# Choose the top N markers only:

N = 50
columnNames = np.unique(markers['cluster'])
topN_markers = pd.DataFrame(index=range(N), columns=columnNames)
for i in range(len(columnNames)):
    topN_temp = np.array(markers[markers['cluster'] == columnNames[i]].iloc[0:N,].iloc[:,0])
    topN_markers[columnNames[i]] = [topN_temp[i].split('.')[0] for i in range(len(topN_temp))]
topN_array = np.array(topN_markers).flatten()    
    
# Let's subset our data to include only marker genes and one set of ROIs across cortical depth in one sample:

subset_rois = [metadata['slide'][i] in rSlides and metadata['Radial_position'][i] in Radial_positions and metadata['AOI_type'][i] in AOI_type for i in range(len(metadata['slide']))]
relevantIDs = metadata['Sample_ID'][subset_rois]
relevantSangerIDs = metadata['Sanger_sampleID'][subset_rois]
counts_subset = counts.loc[:,relevantSangerIDs]
subset_genes = [counts_subset.index[i] in topN_array for i in range(len(counts_subset.index))]
counts_subset = counts_subset.iloc[subset_genes,:]
polioudakis_subset = polioudakis.reindex(np.array(counts_subset.index))

# Now let's identify signals of each cell type in our ROI using Vitalii's model:

mod1 = c2l.LocationModel(
        np.array(polioudakis_subset), np.array(counts_subset).T,
        data_type='float32', n_iter=100000,
        learning_rate=0.0001,
        total_grad_norm_constraint=200,
        verbose=False)

mod1.fit_advi(n=1, method='advi')

mod1.plot_history(1)

mod1.sample_posterior(node='all', n_samples=1000, save_samples=False);

mod1.sample2df(counts_subset.columns, polioudakis_subset.columns, 
               node_name='nUMI_factors')

mod1.compute_expected()
mod1.plot_posterior_mu_vs_data()

results = mod1.spot_factors_df

pickle.dump(results, open("celltypeLocationsPart3.pickle", "wb" ) )

picke.dump(results,)

cellColours = np.array(('green', 'blue', 'purple', 'yellow', 'red'))
celltypes = np.array(('vRG', 'oRG', 'IP', 'ExDp1', 'ExM-U'))

results_normed = np.array(results)
results_normed = (results_normed.T/[sum(results_normed[i,]) for i in range(len(results_normed[:,1]))]).T

figsize(5, 25)
sub_subset = np.array(metadata.loc[subset_rois, 'AOI_type'] == 'HOPXpos')

for i in range(len(celltypes)):
    plt.subplot(5,1, i + 1)
    plt.scatter(sample_info['x'][subset_rois][sub_subset], results_normed[sub_subset,results.columns == celltypes[i]], label = celltypes[i], c = cellColours[i], marker = 'x')
    plt.scatter(sample_info['x'][subset_rois][np.invert(sub_subset)], results_normed[np.invert(sub_subset),results.columns == celltypes[i]], label = celltypes[i], c = cellColours[i], marker = '*')
    plt.title('mRNA Signal for each celltype')
    plt.xlabel('CorticalDepth')
    plt.ylabel('mRNA Proportion')
    plt.legend()
    plt.savefig('RNAContentTypeSpecificROIs.pdf')

### Make a nice plot that shows mRNA content coming from vRG and IP in both EOMES+, HOPX+, rings and background

cellColours = np.array(('green', 'blue', 'purple', 'red', 'green', 'blue', 'purple', 'yellow', 'red', 'green', 'blue', 'purple', 'yellow', 'red', 'green', 'blue', 'purple', 'yellow', 'red', 'green', 'blue', 'purple', 'yellow', 'red'))
celltypes = np.array(('vRG', 'oRG', 'IP', 'ExDp1', 'ExM-U'))
celltypes = np.array(polioudakis.columns)
celltypes = celltypes[np.array((8,13,14,11))]

results_normed = np.array(results)
results_normed = (results_normed.T/[sum(results_normed[i,]) for i in range(len(results_normed[:,1]))]).T

figsize(20, len(celltypes)*5)
for i in range(len(celltypes)):
    plt.subplot(len(celltypes),4, (i)*4 + 1)
    sub_subset = np.array(metadata.loc[subset_rois, 'AOI_type'] == 'EOMESpos')
    plt.scatter(sample_info['x'][subset_rois][sub_subset], results_normed[sub_subset,results.columns == celltypes[i]], label = celltypes[i], c = cellColours[i])
    plt.title('EOMESpos')
    plt.xlabel('CorticalDepth')
    plt.ylabel('mRNA Proportion')
    plt.legend()
    plt.ylim(0, 1) 
    plt.subplot(len(celltypes),4, (i)*4 + 2)
    sub_subset = np.array(metadata.loc[subset_rois, 'AOI_type'] == 'HOPXpos')
    plt.scatter(sample_info['x'][subset_rois][sub_subset], results_normed[sub_subset,results.columns == celltypes[i]], label = celltypes[i], c = cellColours[i])
    plt.title('HOPXpos')
    plt.xlabel('CorticalDepth')
    plt.ylabel('mRNA Proportion')
    plt.legend()
    plt.ylim(0, 1)
    plt.subplot(len(celltypes),4, (i)*4 + 3)
    sub_subset = np.array(metadata.loc[subset_rois, 'AOI_type'] == 'Ring',)
    plt.scatter(sample_info['x'][subset_rois][sub_subset], results_normed[sub_subset,results.columns == celltypes[i]], label = celltypes[i], c = cellColours[i])
    plt.title('Ring')
    plt.xlabel('CorticalDepth')
    plt.ylabel('mRNA Proportion')
    plt.legend()
    plt.ylim(0, 1)
    plt.subplot(len(celltypes),4, (i)*4 + 4)
    sub_subset = np.array(metadata.loc[subset_rois, 'AOI_type'] == 'Residual')
    plt.scatter(sample_info['x'][subset_rois][sub_subset], results_normed[sub_subset,results.columns == celltypes[i]], label = celltypes[i], c = cellColours[i])
    plt.title('Residual')
    plt.xlabel('CorticalDepth')
    plt.ylabel('mRNA Proportion')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig('RNAContentTypeSpecificROIs.pdf')

