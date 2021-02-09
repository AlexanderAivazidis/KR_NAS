from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np

def plot_Locations_1D_scatterPlot(self, x, order = None, polynomial_order = 6, figure_size = (30,30),
                                  saveFig = None, density = True, xlabel = 'x-coordinate', categories = [None]):
    
    # Set figure parameters:
    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    figure(num=None, figsize=figure_size, dpi=80, facecolor='w', edgecolor='k')

    cellColours = np.array(('blue', 'red', 'purple', 'yellow', 'green', 'blue', 'purple', 'yellow', 'red', 'green', 'blue', 'purple', 'yellow', 'red', 'green', 'blue', 'purple', 'yellow', 'red'))
    for i in range(len(self.fact_names)):
        plt.subplot(np.ceil(np.sqrt(len(self.fact_names))),np.ceil(np.sqrt(len(self.fact_names))), i + 1)
        for j in range(len(categories)):
            results_j = np.array(self.spot_factors_df)[order[j],:]
            if density:
                results_j = (results_j.T/[sum(results_j[i,]) for i in range(len(results_j[:,1]))]).T
            y = results_j[:,self.fact_names == self.fact_names[i]][:,0]
            x_j = x[order[j]]
            plt.plot(np.unique(x_j), np.poly1d(np.polyfit(x_j, y, polynomial_order))(np.unique(x_j)), label = categories[j], c = cellColours[j])
            plt.scatter(x_j, y, c = cellColours[j], s = 100)
        plt.xlabel(xlabel)
        if density:
            plt.ylabel('Cell Type Density')
        else:
            plt.ylabel('Cell Type Number')
        plt.legend()
        plt.title(self.fact_names[i])
    plt.tight_layout()
    if saveFig:
        plt.savefig(saveFig)
    plt.show()


# def plot_Locations_1D_dotPlot():
    
#     import scipy as sp

#     markers_genes = np.flipud(np.array(('HES1', 'CRYAB', 'VIM', 'PTN',
#                               'EOMES', 'PPP1R17', 'WNT7B', 'CRYM','TBR1', 'ADRA2A', 'CPLX3', 'NPY', 'GAP43',
#                               'SATB2', 'SYT4', 'SOX5', 'BCL11B',
#                               'ARX', 'DLX2', 'OLIG2',
#                               'CLDN5', 'SPARC','AIF1')))

#     celltypes = np.flipud(np.array(('Ventricular Radial Glia', 'Ventricular Radial Glia', 'Outer Radial Glia', 'Outer Radial Glia',
#                           'Intermediate Progenitor', 'Intermediate Progenitor', 'Subplate Neurons',
#                            'Subplate Neurons','Subplate Neurons', 'Subplate Neurons','Subplate Neurons', 'Subplate Neurons','Subplate Neurons',
#                            'Maturing Excitatory', 'Maturing Excitatory', 'Excitatory Deep Layer', 'Excitatory Deep Layer',
#                            'Interneuron MGE/CGE', 'Interneuron MGE/CGE', 'OPC', 'Endothelial', 'Pericyte', 'Microglia')))

#     subset_w = np.where(np.array([metadata_subset['AOI_type'][i] == 'Geometric' and metadata_subset['slide'][i] == '00MU' for i in range(len(metadata_subset['AOI_type']))]))[0]
#     order = subset_w[np.argsort(metadata_subset['VCDepth'].iloc[subset_w])]

#     normCounts = np.array([X_data[i,:]/sum(X_data[i,:]) for i in range(np.shape(X_data)[0])])*10**6

#     markers_index = np.array([np.where(mod1.genes == markers_genes[i])[0][0] for i in range(len(markers_genes))])

#     indexes = np.unique(celltypes, return_index=True)[1]
#     unique_celltypes = [celltypes[index] for index in sorted(indexes)]
#     counts_z_score = sp.stats.zscore(np.log2(normCounts), axis = 1)

#     genesForPlot = np.repeat(markers_genes,len(order))
#     vcForPlot = np.array([metadata_subset['VCDepth'].iloc[order] for i in range(len(markers_genes))]).flatten()
#     countsForPlot = np.array([sc.stats.zscore(np.log(normCounts[:,markers_index][order,:].T)[i,:]) for i in range(len(markers_genes))])
#     coloursForPlot = np.concatenate([np.repeat(i, sum(celltypes == unique_celltypes[i]) * len(order)) for i in range(len(unique_celltypes))])

#     cmap = matplotlib.cm.get_cmap('tab20')

#     plt.figure(figsize = (12,15))
#     plt.scatter(vcForPlot, genesForPlot, s=((-np.amin(countsForPlot) + countsForPlot)**2)*25, c = cmap(coloursForPlot))
#     plt.xlabel('Cortical Depth')
#     for i in range(len(unique_celltypes)):
#         index = np.where([celltypes[j] == unique_celltypes[i] for j in range(len(celltypes))])[0]
#         index = index[-1]
#         plt.text(1.02, markers_genes[index], celltypes[index], fontsize=20, c = cmap(i))
#     plt.subplots_adjust(left=0.25)

#     #make a legend:
#     pws = [1,-2, -1, 1,2]
#     for pw in pws:
#         plt.scatter([], [], s=((-np.amin(countsForPlot) + pw)**2)*25, c="k",label=str(pw))

#     h, l = plt.gca().get_legend_handles_labels()
#     lgd = plt.legend(h[1:], l[1:], labelspacing=1.2, title="z-score", borderpad=1, 
#                 frameon=True, framealpha=0.6, edgecolor="k", facecolor="w", bbox_to_anchor=(1.55, 0.25))
#     plt.tight_layout()
#     plt.savefig('celltypeMarkers_vs_corticalDepth.pdf', bbox_extra_artists=(lgd,))