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

os.chdir('/home/jovyan/KR_NAS/')

# Let's load all our risk genes:

autism = pd.read_csv('riskGenes/AutismGenes_Satterstrom2020.csv')
autism = np.array(autism['gene'])
brainsize = pd.read_csv('riskGenes/BrainsizeGenes_Grasby2020.csv', skiprows = 2)
subset = np.array(('Average Thickness', 'Banks of the Superior Temporal Sulcus', 'Inferior Temporal', 'Middle Temporal', 'Superior Temporal', 'Total Surface Area'))
brainsize = np.array(brainsize['Gene name'][np.array([brainsize['Phenotype'][i] in subset for i in range(len(brainsize['Phenotype']))])])
ID = pd.read_csv('riskGenes/DDG2P_10_5_2020.csv')
organSpecificity = np.array(ID['organ specificity list'])
ID = ID.loc[~pd.isna(organSpecificity),:]
organSpecificity = np.array(ID['organ specificity list'])
organSpecificity = [organSpecificity[i].split(';') for i in range(len(organSpecificity))]
brainCognition = np.array([sum([organSpecificity[i][j] == 'Brain/Cognition' for j in range(len(organSpecificity[i]))]) > 0 for i in range(len(organSpecificity))])
ID = ID.loc[brainCognition,:]
ID = np.array(ID['gene symbol'])
IQ = pd.read_csv('riskGenes/IQGenes_Savage2018.csv', skiprows=4)
IQ = np.array(IQ.iloc[:507,:]['Gene Name'])

rSlides = np.array(('00MU', '00MV', '00MV-2')) # slide we want to look at #np.array(('00MW')) #
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

subset_rois = [metadata['slide'][i] in rSlides and metadata['Radial_position'][i] in Radial_positions and metadata['AOI_type'][i] == 'Geometric' for i in range(len(metadata['slide']))]
relevantIDs = metadata['Sample_ID'][subset_rois]
relevantSangerIDs = metadata['Sanger_sampleID'][subset_rois]
counts_subset = counts.loc[:,relevantSangerIDs]
properties_subset = properties.loc[np.array(relevantIDs),:]
newLOD_2_subset = properties_subset['NegGeoMean_01']*(properties_subset['NegGeoSD_01']**2) 
newLOD_2        = properties['NegGeoMean_01']*(properties['NegGeoSD_01']**2)
corticalDepth = np.array(metadata['VCDepth'])[subset_rois]

# Good genes:

goodGenes = np.array([counts_subset.iloc[:,i]  > newLOD_2_subset[i] for i in range(len(newLOD_2_subset))])
goodGenes = np.array([sum(goodGenes[:,i]) > 2 for i in range(len(goodGenes[1,:]))])

# Plot the number of genes detected from each set as a function of cortical depth:

autism_index = np.where([counts_subset.index[i] in autism for i in range(len(counts_subset.index))])[0]
brainsize_index = np.where([counts_subset.index[i] in brainsize for i in range(len(counts_subset.index))])[0]
ID_index = np.where([counts_subset.index[i] in ID for i in range(len(counts_subset.index))])[0]
IQ_index = np.where([counts_subset.index[i] in IQ for i in range(len(counts_subset.index))])[0]

# Raw:

autism_detected = np.array([sum(counts_subset.iloc[autism_index,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])
brainsize_detected = np.array([sum(counts_subset.iloc[brainsize_index,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])
ID_detected = np.array([sum(counts_subset.iloc[ID_index,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])
IQ_detected = np.array([sum(counts_subset.iloc[IQ_index,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])
All_detected = np.array([sum(counts_subset.iloc[:,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])

plt.scatter(corticalDepth, autism_detected/len(autism), label = 'Autism')
plt.scatter(corticalDepth, brainsize_detected/len(brainsize), label = 'Brainsize')
plt.scatter(corticalDepth, ID_detected/len(ID), label = 'All NDDs')
plt.scatter(corticalDepth, IQ_detected/len(IQ), label = 'IQ')
plt.scatter(corticalDepth, All_detected/18190, label = 'All Genes', c = 'grey')
plt.xlabel('Cortical Depth')
plt.ylabel('Proportion of Gene Set Detected')
plt.legend()
plt.ylim(-0.1,1)
plt.savefig('GeneSetDetectionNew.pdf')
plt.show()

# Normalized:

autism_detected = np.array([sum(counts_subset.iloc[autism_index,i] > newLOD_2_subset[i])/sum(counts_subset.iloc[:,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])
brainsize_detected = np.array([sum(counts_subset.iloc[brainsize_index,i] > newLOD_2_subset[i])/sum(counts_subset.iloc[:,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])
ID_detected = np.array([sum(counts_subset.iloc[ID_index,i] > newLOD_2_subset[i])/sum(counts_subset.iloc[:,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])
IQ_detected = np.array([sum(counts_subset.iloc[IQ_index,i] > newLOD_2_subset[i])/sum(counts_subset.iloc[:,i] > newLOD_2_subset[i]) for i in range(len(newLOD_2_subset))])

plt.scatter(corticalDepth, autism_detected, label = 'Autism')
plt.scatter(corticalDepth, brainsize_detected, label = 'Brainsize')
plt.scatter(corticalDepth, ID_detected, label = 'ID')
plt.scatter(corticalDepth, IQ_detected, label = 'IQ')
plt.xlabel('Cortical Depth')
plt.ylabel('Proportion of Detected Genes')
plt.legend()
plt.show()

# Plot the summed scaled expression of gene sets as a function of cortical depth:

counts_subset = counts_subset.loc[goodGenes,:]

autism_index = np.where([counts_subset.index[i] in autism for i in range(len(counts_subset.index))])[0]
brainsize_index = np.where([counts_subset.index[i] in brainsize for i in range(len(counts_subset.index))])[0]
ID_index = np.where([counts_subset.index[i] in ID for i in range(len(counts_subset.index))])[0]
IQ_index = np.where([counts_subset.index[i] in IQ for i in range(len(counts_subset.index))])[0]


# Raw:

autism_detected = np.array([sum(np.log2(counts_subset.iloc[autism_index,i])) for i in range(len(newLOD_2_subset))])
brainsize_detected = np.array([sum(np.log2(counts_subset.iloc[brainsize_index,i])) for i in range(len(newLOD_2_subset))])
ID_detected = np.array([sum(np.log2(counts_subset.iloc[ID_index,i])) for i in range(len(newLOD_2_subset))])
IQ_detected = np.array([sum(np.log2(counts_subset.iloc[IQ_index,i])) for i in range(len(newLOD_2_subset))])

plt.scatter(corticalDepth, autism_detected, label = 'Autism')
plt.scatter(corticalDepth, brainsize_detected, label = 'Brainsize')
plt.scatter(corticalDepth, ID_detected, label = 'ID')
plt.scatter(corticalDepth, IQ_detected, label = 'IQ')
plt.xlabel('Cortical Depth')
plt.ylabel('Log2 Counts Sum')
plt.legend()
plt.show()

# Normalized:

import scipy as sc
norm = np.array([counts_subset.iloc[:,i]/sum(counts_subset.iloc[:,i]) for i in range(len(counts_subset.iloc[1,:]))]).T
counts_z_score = sc.stats.zscore(np.log2(norm+1), axis = 1)

autism_detected = np.array([sum(counts_z_score[autism_index,i])/len(autism) for i in range(len(newLOD_2_subset))])
brainsize_detected = np.array([sum(counts_z_score[brainsize_index,i])/len(brainsize) for i in range(len(newLOD_2_subset))])
ID_detected = np.array([sum(counts_z_score[ID_index,i])/len(ID) for i in range(len(newLOD_2_subset))])
IQ_detected = np.array([sum(counts_z_score[IQ_index,i])/len(IQ) for i in range(len(newLOD_2_subset))])

All_detected = np.array([sum(counts_z_score[:,i])/18119 for i in range(len(newLOD_2_subset))])

plt.scatter(corticalDepth, brainsize_detected, label = 'Brainsize')
plt.scatter(corticalDepth, IQ_detected, label = 'IQ')
plt.scatter(corticalDepth, ID_detected, label = 'All NDDs')
plt.scatter(corticalDepth, All_detected, label = 'All Genes', c = 'grey')
plt.scatter(corticalDepth, autism_detected, label = 'Autism')
plt.xlabel('Cortical Depth')
plt.ylabel('Mean Z-score of Log2(Nomalized Counts)')
plt.ylim(-1,1)
plt.legend()
plt.savefig('GeneSetTrajectoriesNew.pdf')