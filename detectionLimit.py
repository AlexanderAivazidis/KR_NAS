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

rSlides = np.array(('00MU', '00MV')) # slide we want to look at
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
markers.iloc[:,0] = [markers.iloc[:,0][i].split('.')[0] for i in range(np.shape(markers)[0])]
genes = polioudakis.iloc[:,0]
polioudakis = polioudakis.drop('Unnamed: 0',1)
polioudakis = polioudakis.rename(index=genes)

subset_rois = [metadata['slide'][i] in rSlides and metadata['Radial_position'][i] == 4 and metadata['AOI_type'][i] == 'Geometric' for i in range(len(metadata['slide']))]
relevantIDs = metadata['Sample_ID'][subset_rois]
relevantSangerIDs = metadata['Sanger_sampleID'][subset_rois]
counts_subset = counts.loc[:,relevantSangerIDs]
properties_subset = properties.loc[np.array(relevantIDs),:]
newLOD_2_subset = properties_subset['NegGeoMean_01']*(properties_subset['NegGeoSD_01']**2) 
newLOD_2        = properties['NegGeoMean_01']*(properties['NegGeoSD_01']**2)

markers = markers.loc[markers['p_val_adj'] < 0.05,:]

vRG_genes = np.array(markers[markers['cluster'] == "vRG"].iloc[:,0]) 
vRG_genes = vRG_genes[[vRG_genes[i] in np.array(counts_subset.index) for i in range(len(vRG_genes))]]

# Let's check how many genes we detect above LOD as a function of AOI size or number of nuclei:

genesDetected = [np.sum(counts_subset.iloc[:,i]  > newLOD_2_subset[i]) for i in range(4)]
markersDetected = [np.sum(counts_subset.loc[vRG_genes,:].iloc[:,i]  > newLOD_2_subset[i]) for i in range(4)]

figsize(4, 4)
plt.scatter(properties_subset['nuclei'], genesDetected, c = 'blue', label = 'All Genes')
plt.scatter(properties_subset['nuclei'], markersDetected, c = 'orange', label = 'vRG Markers')
plt.ylim(0,7000)
plt.xlabel('Number of nuclei')
plt.ylabel('Genes expressed above level of detection')
plt.legend()
plt.tight_layout()
plt.savefig('otherPlots/DetectedGenes1.pdf')

plt.scatter(properties_subset['nuclei'], genesDetected, c = 'blue')
plt.scatter(properties_subset['roi_dimension'], genesDetected, c = 'orange')
plt.xlabel('Number of Nuclei')
plt.ylabel('Genes above LOD_2')
plt.savefig('otherPlots/DetectedGenes2.pdf')

# Let's check how many genes we detect above LOD as a function of AOI size or number of nuclei (this time for all geometric AOIs):

genesDetected_All = [np.sum(counts.iloc[:,i]  > newLOD_2[i]) for i in range(np.shape(counts)[1])]

plt.scatter(properties['nuclei'].loc[np.array(metadata['AOI_type'] == 'Geometric')], 
            np.array(genesDetected_All)[np.array(metadata['AOI_type'] == 'Geometric')], c = 'blue')
plt.xlabel('Number of Nuclei')
plt.ylabel('Genes expressed above level of detection')
plt.tight_layout()
plt.savefig('otherPlots/DetectedGenes1.pdf')

# Let's plot the average UMI of each vRG gene in the Polioudakis data vs. the average UMI's in different sized ROI's:

figsize(20, 5)
for i in range(4):
    plt.subplot(1,4, i + 1)
    plt.scatter(counts_subset.iloc[[counts_subset.index[i] in vRG_genes for i in range(len(counts_subset.index))], 0]/properties_subset['nuclei'][i], counts_subset.iloc[[counts_subset.index[i] in vRG_genes for i in range(len(counts_subset.index))], i]/properties_subset['nuclei'][i])
    plt.title('ROI with ' + str(int(properties_subset['nuclei'][i])) + ' nuclei vs. 95 nuclei')
    #plt.ylim(0,4)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('UMI count/nuclei in GeoMx data')
    plt.ylabel('UMI count/nuclei in GeoMx data')
    plt.tight_layout()
    plt.savefig('otherPlots/Sensitivity1.pdf')
plt.show()

figsize(20, 5)
for i in range(4):
    plt.subplot(1,4, i+1)
    plt.scatter(polioudakis.loc[vRG_genes,'vRG'], counts_subset.iloc[[counts_subset.index[i] in vRG_genes for i in range(len(counts_subset.index))], i]/properties_subset['nuclei'][i])
    plt.title('ROI with ' + str(int(properties_subset['nuclei'][i])) + ' nuclei vs reference')
    plt.ylim(0,100)
    plt.xlim(0,100)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Mean UMI count in Polioudakis et al. 2019 vRG cluster')
    plt.ylabel('UMI count/nuclei in GeoMx data')
    plt.tight_layout()
    plt.savefig('otherPlots/Sensitivity2.pdf')
plt.show()

# Let's plot the average UMI of each vRG gene in the Polioudakis data vs. the average UMI's in different sized ROI's: