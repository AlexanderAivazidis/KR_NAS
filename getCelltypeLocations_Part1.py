## This scripts gets genes with spatial structures in the two 19pcw replicates using SpatialDE and makes some nice plots
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
os.environ["THEANO_FLAGS"] = 'device=cpu,floatX=' + data_type + ',force_device=True'
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
from matplotlib import rc

os.chdir('/home/jovyan/KR_NAS/')

rSlides = np.array(('00MU', '00MV')) # slide we want to look at
AOI_type = ['Geometric'] #np.array(('HOPXpos', 'EOMESpos', 'Residual', 'Ring', 'Geometric'))
Radial_positions = np.array((2))

# Let's load the data and metadata from our Nanostring experiment:
counts = pd.read_table('Sanger_288ROIs_TargetCountMatrix.txt')
genes = counts['TargetName']
counts = counts.drop('TargetName',1)
counts = counts.rename(index=genes)
counts = counts.astype('int')
counts = counts[np.array([sum(counts.iloc[i,:]) for i in range(len(counts.iloc[:,0]))]) > 3 * 288]
metadata = pd.read_csv('NanoString sequencing all annotations 2020.02.10.csv')
metadata = metadata.iloc[0:289,]
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
        data_type='float32', n_iter=50000,
        learning_rate=0.0001,
        total_grad_norm_constraint=200,
        verbose=False)

mod1.sample_prior()
mod1.plot_prior_vs_data()

mod1.fit_advi(n=1, method='advi')

mod1.plot_history(1)

mod1.sample_posterior(node='all', n_samples=1000, save_samples=False);

mod1.sample2df(counts_subset.columns, polioudakis_subset.columns, 
               node_name='nUMI_factors')

mod1.compute_expected()
mod1.plot_posterior_mu_vs_data()

# Save the results:
results = mod1.spot_factors_df

results_normed = np.array(results)
results_normed = (results_normed.T/[sum(results_normed[i,]) for i in range(len(results_normed[:,1]))]).T

# Plot the results as a stacked bar plot:
cellColours = np.array(('green', 'blue', 'purple', 'yellow', 'red'))
celltypes = np.array(('vRG', 'oRG', 'IP', 'ExDp1', 'ExM-U'))
results_normed = np.array(results)
results_normed = (results_normed - results_normed.min(0)) / results_normed.ptp(0)
figsize(5, 25)
for i in range(len(celltypes)):
    plt.subplot(5,1, i + 1)
    plt.scatter(sample_info['x'][subset_rois], results_normed[:,results.columns == celltypes[i]], label = celltypes[i], c = cellColours[i])
    plt.title('mRNA Signal for each celltype')
    plt.xlabel('CorticalDepth')
    plt.ylabel('Normalized mRNA')
    plt.legend()
    plt.savefig('CellLocations.pdf')


