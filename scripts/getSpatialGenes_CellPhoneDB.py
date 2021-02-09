## Get the number of genes above LOD as a function of nuclei number:
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
import NaiveDE
import SpatialDE
import statsmodels.stats.multitest as multi
import scipy.stats as ss

os.chdir('/home/jovyan/KR_NAS/')
rSlides = ('00MU', '00MV', '00MV-2')
AOI_type = 'Geometric'
Radial_position = 2

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
genes = polioudakis.iloc[:,0]
polioudakis = polioudakis.drop('Unnamed: 0',1)
polioudakis = polioudakis.rename(index=genes)

subset_rois = [metadata['age'][i] == '19pcw' and metadata['slide'][i] in rSlides and metadata['Radial_position'][i] == Radial_position and metadata['AOI_type'][i] == AOI_type for i in range(len(metadata['slide']))]
relevantIDs = metadata['Sample_ID'][subset_rois]
relevantSangerIDs = metadata['Sanger_sampleID'][subset_rois]
counts_subset = counts.loc[:,relevantSangerIDs]
properties_subset = properties.loc[np.array(relevantIDs),:]
newLOD_2_subset = properties_subset['NegGeoMean_01']*(properties_subset['NegGeoSD_01']**2) 

# Load all the receptor-ligand pairs from CellPhoneDB:

cellPhoneDB_geneInput = pd.read_csv('/home/jovyan/data/CellPhoneDB/gene_input.csv')
cellPhoneDB_genes = unique(cellPhoneDB_geneInput.iloc[:,0])
cellPhoneDB_ensembl = cellPhoneDB_geneInput.iloc[:,3]
cellPhoneDB_interactions = pd.read_csv('/home/jovyan/data/CellPhoneDB/interaction_curated.csv')
cellPhoneDB_interactions = cellPhoneDB_interactions.fillna('')
cellPhoneDB_proteins = np.unique(np.array(([cellPhoneDB_interactions.iloc[i,3].split('_')[0] for i in range(len(cellPhoneDB_interactions.iloc[:,3]))],
                                          [cellPhoneDB_interactions.iloc[i,4].split('_')[0] for i in range(len(cellPhoneDB_interactions.iloc[:,3]))])))
lr_geoMx = counts_subset.index[[counts_subset.index[i] in np.array(cellPhoneDB_genes) for i in range(len(counts_subset.index))]]

norm_expr = NaiveDE.stabilize(counts.T).T

resid_expr = NaiveDE.regress_out(sample_info, norm_expr, 'np.log(Q3_counts)').T
resid_expr_subset = resid_expr.loc[subset_rois,lr_geoMx]

X = sample_info[['x', 'y']][subset_rois]
results = SpatialDE.run(X, resid_expr_subset)

results = results.sort_values('qval')
genes_ranked = np.array(results['g'])
genes_significant = sum(multi.multipletests(results['pval'], method = 'fdr_bh')[1] < 0.05)

results['FDR'] = multi.multipletests(results['pval'], method = 'fdr_bh')[1]

# Save results:

colours = np.repeat('blue', sum(subset_rois))
colours[metadata['slide'][subset_rois] == '00MU'] = 'red'
figsize(15, 10)
for i in range(16):
    plt.subplot(4,4, i + 1)
    plt.scatter(sample_info['x'][subset_rois], np.array(resid_expr_subset[genes_ranked[i]]), c=colours);
    plt.title(genes_ranked[i])
    plt.xlabel('CorticalDepth')
    plt.ylabel('Resid. Expr.')
    plt.tight_layout()
    plt.savefig('CellPhoneDB_SpatialDE_genes.png') 
plt.show()