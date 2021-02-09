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

rSlides = np.array(('00MU', '00MV'))
AOI_type = 'Geometric'
threshold = 0.5

# Load relevant results:        

results = pickle.load(open("resultLists/resultListnormMethdQ3Slides" + "".join(rSlides) + ".pickle", "rb" ))    

# Find genes that are actually expressed in those ROIs above LOD:        

print(rSlides)
subset = [metadata['slide'][i] in rSlides and metadata['AOI_type'][i] in AOI_type and metadata['Radial_position'][i] == 2 for i in range(len(metadata['slide']))]
noSamples = sum(subset)
relevantIDs = metadata['Sample_ID'][subset]
relevantSangerIDs = metadata['Sanger_sampleID'][subset]
counts_subset = counts.loc[:,relevantSangerIDs]
properties_subset = properties.loc[relevantIDs,:]    
geneCounts = [sum([counts_subset.iloc[i,j] > properties_subset.iloc[j,19] for j in range(noSamples)])/noSamples for i in range(np.shape(counts_subset)[0])]

goodGenes = counts_subset.index[np.array(geneCounts) > 0]

# Recalculate FDR based on those genes only:

results_subset = results[[results['g'].iloc[i] in goodGenes for i in range(len(results['g']))]]
results_subset['FDR'] = np.array(multi.multipletests(np.array(results_subset['pval']), method = 'fdr_bh')[1])

markers = pd.read_csv('/home/jovyan/data/fetalBrain/Polioudakis/clusterMarkers.csv')

polioudakis = pd.read_csv('/home/jovyan/data/fetalBrain/Polioudakis/cellStateMatrix.csv')
polioudakis_all = np.array(polioudakis.iloc[:,0])
polioudakis.index = polioudakis_all
polioudakis_top = np.array(markers.iloc[:,0][markers['p_val_adj'] < 0.05])

topGenes = results_subset['g'][results_subset['FDR'] < 0.05]

commonGenes = np.intersect1d(goodGenes, polioudakis_all)

topGenes_common = topGenes[[topGenes.iloc[i] in commonGenes for i in range(len(topGenes))]]
polioudakis_common = polioudakis_top[[polioudakis_top[i] in commonGenes for i in range(len(polioudakis_top))]]

newGenes = np.setdiff1d(topGenes_common, polioudakis_common)

# How many genes were detected in both our and polioudakis data?

print(len(commonGenes))

# How many of those genes show up as variable with our or their method?
print(len(topGenes_common))
print(len(polioudakis_common))

# How many of those genes are exclusive to our data?

print(len(newGenes))

# Subset results:

results_newGenes = results[[results['g'].iloc[i] in newGenes for i in range(len(results['g']))]]

# Plot results:

norm_expr = NaiveDE.stabilize(counts.T).T

counts_Q3 = np.array(NaiveDE.regress_out(sample_info, norm_expr, 'np.log(Q3_counts)').T).T

genes = np.array(results_newGenes['g'])[0:10]

figure(num=None, figsize=(5, len(genes)*5), dpi=80, facecolor='w', edgecolor='k')
for i in range(len(genes)):
    plt.subplot(len(genes),1, i + 1)
    plt.scatter(sample_info['x'][subset], counts_Q3[np.where(np.array(counts.index) == genes[i])[0][0],subset], label = genes[i])
    plt.title(genes[i])
    plt.xlabel('CorticalDepth')
    plt.ylabel('Normalized Count')
    plt.legend()
    
genes_plot = ["PNRC1", "BTG1", "CHL1"]
celltypes_plot = polioudakis.columns[1:]
geneExpression = polioudakis.loc[genes_plot ,celltypes_plot]
fig, ax = plt.subplots()
im = ax.imshow(geneExpression)
# We want to show all ticks...
ax.set_xticks(np.arange(len(celltypes_plot)))
ax.set_yticks(np.arange(len(genes_plot)))
# ... and label them with the respective list entries
ax.set_xticklabels(celltypes_plot)
ax.set_yticklabels(genes_plot)
# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(genes_plot)):
    for j in range(len(celltypes_plot)):
        text = ax.text(j, i, np.round(geneExpression.iloc[i, j],1),
                       ha="center", va="center", color="w")
ax.set_title("Expression in Polioudakis 2019 data")
fig.tight_layout()
plt.savefig('ExpressionInPolioudakis2019.pdf')
plt.show()