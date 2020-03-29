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

rSlides_array = np.array((('00MU', '00MV', '00MV-2'),('00MU', '00MV', '00MV-2'),('00MU', '00MV', '00MV-2'))) # slide we want to look at
AOI_type_array = np.array(('Geometric', 'EOMESpos', 'HOPXpos'))
Radial_position_array = np.array(((4,10),(1,3),(1,3)))
comparison_array = np.array(('vRG', 'IP', 'oRG'))

for index in range(len(rSlides)):
    
    print(index)
    rSlides = rSlides_array[index]
    AOI_type = AOI_type_array[index]
    Radial_position = Radial_position_array[index]
    comparison = comparison_array[index]
    
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

    subset_rois = [metadata['age'][i] == '19pcw' and metadata['slide'][i] in rSlides and metadata['Radial_position'][i] in Radial_position and metadata['AOI_type'][i] == AOI_type for i in range(len(metadata['slide']))]
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

    # Show total receptors/ligands detected by GeoMx and Polioudakis scRNAseq, as well as exclusive ones both for the oRG and IPs:

    lr_db = len(cellPhoneDB_genes)
    lr_geoMx = sum([counts_subset.index[i] in np.array(cellPhoneDB_genes) for i in range(len(counts_subset.index))])
    lr_polioudakis = sum([polioudakis.index[i] in np.array(cellPhoneDB_genes) or polioudakis.index[i] in np.array(cellPhoneDB_ensembl) for i in range(len(polioudakis.index))])
    lr_both = counts_subset.index[[counts_subset.index[i] in np.array(cellPhoneDB_genes) and counts_subset.index[i] in polioudakis.index for i in range(len(counts_subset.index))]]

    detected_geoMx = [np.sum(counts_subset.loc[lr_both].iloc[:,i]  > newLOD_2_subset[i]) for i in range(len(counts_subset.iloc[0,:]))]
    detected_polioudakis = [np.sum(polioudakis.loc[np.array(lr_both),comparison] > threshold) for threshold in np.array((1,0.1,0.01,0.001))]

    # Simple plot to show number detected:

    figsize(6, 4)
    plt.scatter(properties_subset['nuclei'], detected_geoMx, c = 'blue')
    plt.xlabel('Number of Nuclei')
    plt.ylabel('Genes expressed above level of detection')
#     plt.ylim((0,180))
#     plt.xlim((0,100))
    plt.tight_layout()
    plt.savefig('otherPlots/DetectedDBGenes1' + AOI_type + '.pdf')
    plt.show()
    
    figsize(6, 4)
    plt.scatter(np.array((1,0.1,0.01,0.001)), detected_polioudakis, c = 'blue')
    plt.xlabel('Mean UMI Threshold')
    plt.ylabel('Genes expressed above level of detection')
    #plt.xlim((0,100))
    plt.tight_layout()
    plt.savefig('otherPlots/DetectedDBGenes2' + AOI_type + '.pdf')
    plt.show()
    
    # Simple plot to show overlap:      
    overlap = np.empty((len(properties_subset['nuclei']),4))
    for i in range(len(properties_subset['nuclei'])):
        for j in range(4):
            overlap[i,j] = len(np.intersect1d(lr_both[counts_subset.loc[lr_both].iloc[:,i]  > newLOD_2_subset[i]],
                    lr_both[polioudakis.loc[np.array(lr_both),comparison]  > np.array((1,0.1,0.01,0.001))[j]]))/detected_polioudakis[j]  

    figsize(12, 8)
    numberOfNuclei = properties_subset['nuclei']
    order = np.array(np.argsort(numberOfNuclei))
    numberOfNuclei = numberOfNuclei[order]
    overlap = overlap[order,:]
    UMIthresholds = np.array((1,0.1,0.01,0.001))
    fig, ax = plt.subplots()
    im = ax.imshow(overlap)
    # We want to show all ticks...
    ax.set_yticks(np.arange(len(numberOfNuclei)))
    ax.set_xticks(np.arange(len(UMIthresholds)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(UMIthresholds)
    ax.set_yticklabels(numberOfNuclei)
    ax.set_ylabel('Number Of Nuclei')
    ax.set_xlabel('Mean UMI threshold')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(numberOfNuclei)):
        for j in range(len(UMIthresholds)):
            text = ax.text(j, i, np.round(overlap[i,j],2),
                           ha="center", va="center", color="w")
    ax.set_title("Overlap")
    fig.tight_layout()
    plt.savefig('otherPlots/OverlapVZ' + AOI_type + '.pdf')
    plt.show()

