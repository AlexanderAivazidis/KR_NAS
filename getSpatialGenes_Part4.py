# Make plots of genes with most spatial structure in different ROIs and compare to the other ROIs at the same cortical depth:

import pickle
import os
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as multi
import NaiveDE
import SpatialDE
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure

os.chdir('/home/jovyan/KR_NAS/')

# Let's load the data and metadata:
counts = pd.read_table('/home/jovyan/KR_NAS/Sanger_288ROIs_TargetCountMatrix.txt')
genes = counts['TargetName']
counts = counts.drop('TargetName',1)
counts = counts.rename(index=genes)
counts = counts.astype('int')
counts = counts[np.array([sum(counts.iloc[i,:]) for i in range(len(counts.iloc[:,0]))]) > 3 * 288]
metadata = pd.read_csv('/home/jovyan/KR_NAS/NanoString sequencing all annotations 2020.02.10.csv')
metadata = metadata.iloc[0:286,]
metadata = metadata.rename(index=metadata['Sanger_sampleID'])
counts = counts.iloc[:,[counts.columns[i] in metadata.index for i in range(len(counts.columns))]]
metadata = metadata.reindex(np.array(counts.columns))    
properties = pd.read_table('/home/jovyan/KR_NAS/Sanger_288ROIs_SegmentProperties.txt')
properties = properties.rename(index=properties['DSP_Sample_ID'])
properties = properties.reindex(np.array(metadata['Sample_ID']))
columnNames = ('x', 'y', 'total_counts', 'Q3_counts')
sample_info = pd.DataFrame(index=metadata['Sample_ID'], columns=columnNames)
sample_info['x'] = np.array(metadata['VCDepth'])
sample_info['y'] = np.array(metadata['Radial_position'])
sample_info['total_counts'] = [sum(counts.iloc[:,i]) for i in range(len(counts.iloc[1,:]))] 
sample_info['Q3_counts'] = [sum(np.sort(counts.iloc[:,i])[int(np.round(0.5*len(counts.iloc[:,i]))):int(np.round(0.75*len(counts.iloc[:,i])))]) for i in range(len(counts.iloc[1,:]))] 

rSlides = np.array(('00MU', '00MV', '00MV-2'))
AOI_type_Array = np.array(('EOMESpos', 'HOPXpos', 'Ring', 'Residual'))
threshold = 0.5

numberOfTopsGenes = []
topGenes = pd.DataFrame(index=range(100), columns=AOI_type_Array)

for AOI_type in AOI_type_Array:

    # Load relevant results:        
        
    results = pickle.load(open("resultLists/resultListnormMethdQ3Slides" + "".join(rSlides) + "AOI_type" + AOI_type + ".pickle", "rb" ))    

    # Find genes that are actually expressed in those ROIs above LOD:        

    print(rSlides)
    subset = [metadata['slide'][i] in rSlides and metadata['AOI_type'][i] in AOI_type for i in range(len(metadata['slide']))]
    noSamples = sum(subset)
    relevantIDs = metadata['Sample_ID'][subset]
    relevantSangerIDs = metadata['Sanger_sampleID'][subset]
    counts_subset = counts.loc[:,relevantSangerIDs]
    properties_subset = properties.loc[relevantIDs,:]    
    geneCounts = [sum([counts_subset.iloc[i,j] > properties_subset.iloc[j,19] for j in range(noSamples)])/noSamples for i in range(np.shape(counts_subset)[0])]

    # Recalculate FDR based on those genes only:

    results_subset = results[[results['g'].iloc[i] in goodGenes for i in range(len(results['g']))]]
    results_subset['FDR'] = np.array(multi.multipletests(np.array(results_subset['pval']), method = 'fdr_bh')[1])
    topGenes[AOI_type] = np.array(results_subset['g'])[0:100]
    numberOfTopsGenes.append(sum(results_subset['FDR'] < 0.05))
    
    norm_expr = NaiveDE.stabilize(counts.T).T

    counts_Q3 = np.array(NaiveDE.regress_out(sample_info, norm_expr, 'np.log(Q3_counts)').T).T
    
    AOI_type = 'EOMESpos'
    genes = topGenes[AOI_type]
    genes = genes.iloc[0:10]
    
    figure(num=None, figsize=(20, len(genes)*5), dpi=80, facecolor='w', edgecolor='k')
    for i in range(len(genes)):
        plt.subplot(len(genes),1, i + 1)
        sub_subset = np.array(metadata.loc[:, 'AOI_type'] == 'EOMESpos')
        plt.scatter(sample_info['x'][sub_subset], counts_Q3[np.where(np.array(counts.index) == genes.iloc[i]),sub_subset], label = genes.iloc[i])
        plt.title(genes.iloc[i])
        plt.xlabel('CorticalDepth')
        plt.ylabel('Normalized Count')
        plt.legend()



