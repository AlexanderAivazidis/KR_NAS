## This scripts gets genes with spatial structures in the two 19pcw replicates using SpatialDE and makes some nice plots

%pylab inline
import pandas as pd
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
import NaiveDE
import SpatialDE
import statsmodels.stats.multitest as multi
import scipy.stats as ss
import pickle as pickle

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

relevantSlides = np.array(('00MU', '00MV', ('00MU', '00MV')))
normalizationMethods = ['Q3']
qValueS = np.array((0.000001, 0.00001, 0.0001, 0.001, 0.01))
numberOfGroups = np.array((4,6,8,10))
lengthScales = np.array((0.1, 0.2, 0.4))
topGenesArray = np.array((10,100,1000))

rSlides = ('00MU', '00MV')
normMethod = 'Q3'

# Make a humongous set of for loops that generates figures according to many different parameters, so that we can choose the one we like the most:

for rSlides in relevantSlides:
    
    print(rSlides)
    subset = [metadata['slide'][i] in rSlides and metadata['Radial_position'][i] == 2 for i in range(len(metadata['slide']))]
    relevantIDs = metadata['Sample_ID'][subset]
    relevantSangerIDs = metadata['Sanger_sampleID'][subset]
    norm_expr = NaiveDE.stabilize(counts.T).T

    for normMethod in normalizationMethods:
        print(normMethod)
        if normMethod == 'totalCounts':
            resid_expr = NaiveDE.regress_out(sample_info, norm_expr, 'np.log(total_counts)').T
            resid_expr_subset = resid_expr[subset]

            X = sample_info[['x', 'y']][subset]
            results = SpatialDE.run(X, resid_expr_subset)

            results = results.sort_values('qval')
            genes_ranked = np.array(results['g'])
            genes_significant = sum(multi.multipletests(results['pval'], method = 'fdr_bh')[1] < 0.05)

        ## Now we try Q3 counts normalization:
        if normMethod == 'Q3':
            resid_expr = NaiveDE.regress_out(sample_info, norm_expr, 'np.log(Q3_counts)').T
            resid_expr_subset = resid_expr[subset]

            X = sample_info[['x', 'y']][subset]
            results = SpatialDE.run(X, resid_expr_subset)

            results = results.sort_values('qval')
            genes_ranked = np.array(results['g'])
            genes_significant = sum(multi.multipletests(results['pval'], method = 'fdr_bh')[1] < 0.05)
        
        results['FDR'] = multi.multipletests(results['pval'], method = 'fdr_bh')[1]
        
        # Save results:
        
        pickle.dump(results, open('resultLists/' + 'resultList' + 'normMethd' + normMethod + 'Slides' + ''.join(rSlides) + ".pickle", "wb" ) )
        results.to_csv('resultLists/''resultList' + 'normMethd' + normMethod + 'Slides' + ''.join(rSlides) + ".csv")
        
        colours = np.repeat('blue', sum(subset))
        colours[metadata['slide'][subset] == '00MU'] = 'red'
        figsize(30, 20)
        for i in range(100):
            plt.subplot(10,10, i + 1)
            plt.scatter(sample_info['x'][subset], np.array(resid_expr_subset[genes_ranked[i]]), c=colours);
            plt.title(genes_ranked[i])
            plt.xlabel('CorticalDepth')
            plt.ylabel('Resid. Expr.')
            plt.tight_layout()
            plt.savefig('topSpatialGenes/top100SpatialGenes' + 'normMethd' + normMethod + 'Slides' + ''.join(rSlides) + '.png') 
        plt.show()
            
        favouriteGenes = np.array(('VIM', 'HOPX','EOMES', 'NES', 'PAX6', 'SLC1A3', 'GFAP', 'PTPRZ1', 'HES1', 'HES5', 'CDH2', 'SPARCL1', 'TNC', 'PDGFRA', 'NTRK2', 'NEUROD6', 'SOX2', 'DCX', 'MAP2', 'SYP', 'SOX10', 'GAP43', 'MEF2C', 'NPY', 'GJA3', 'APOL3', 'RELN', 'NCAM2', 'STMN2', 'MAP1B', 'TUBB2B', 'TUBA1C,TUBA1A,TUBA1B', 'SATB2', 'DACT1', 'SYT4', 'TUBB3', 'AIF1', 'SOX5', 'BCL11B', 'SATB2'))
        
        favouriteGenes = np.array(('HES1', 'TNC', 'TUBB3', 'NTRK2'))
        
        favouriteGenes = np.array(('LMO3', 'CRYM', 'FBXW7', 'TLE4', 'LMO7', 'PPP1R1B',   'MEG3',    'KCTD12',    'PDE1A',     'ZNF385D',   'NRCAM',   'GRIN2B',    'SYT6',      'SORCS1',    'ST18',      'CELF4',   'SEMA3E',    'ANK3',    'WASF1',   'SCN2A',  'ATP1B1',  'TRPM3',     'VSNL1',     'SCD5' ,  
'TMEM178A',  'NUAK1',     'LPL',       'ADCY1',     'GPR22',     'NR4A2',     'KHDRBS3',   'CHL1' ,   'PLEKHH2' ,  'EPB41L4A',  'SOBP' ,   'MEF2C',     'GRIA2' ,  'ITSN1',   'KCNMA1',    'TP53I11',   'PCP4',      'SMARCA2', 'AKAP7' ,    'MGLL' ,   'GAS7',      'NEGR1' ,    'PPM1L',   'PEX5L'))
        
        favouriteGenes = favouriteGenes[[favouriteGenes[i] in resid_expr_subset.columns for i in range(len(favouriteGenes))]]
        
        colours = np.repeat('red', sum(subset))
        #colours[metadata['slide'][subset] == '00MU'] = 'red'
        figsize(5, 50)
        for i in range(len(favouriteGenes)):
            plt.subplot(23,2, i + 1)
            plt.scatter(sample_info['x'][subset], np.array(resid_expr_subset[favouriteGenes[i]]), c=colours);
            plt.title(favouriteGenes[i])
            plt.xlabel('CorticalDepth')
            plt.ylabel('Resi. Expr.')
            plt.tight_layout()
            plt.savefig('favouriteGenes/KennysFavouriteGenes' + 'normMethd' + normMethod +  'Slides' + ''.join(rSlides) + '.pdf') 
        plt.show()
        ## Let's group the genes into typical spatial patterns:

        markerGenes = np.array(('CRYAB', 'HOPX', 'SOX2', 'HES1', 'PAX6', 'VIM', 'EOMES', 'NEUROD6', 'STMN2', 'SOX5', 'TBR1', 'BCL11B', 'SATB2', 'CALB2', 'SST', 'DLX2', 'DLX1', 'OLIG2', 'OLIG1', 'CLDN5', 'ITM2A', 'RGS5', 'CX3CR1', 'AIF1'))
        
        for n_patterns in numberOfGroups:
            print(n_patterns)
            for cutoff in qValueS:
                print(cutoff)
                for l in lengthScales:
                    print(l)
                    sign_results = results.query('qval < ' + str(cutoff))
                    if np.shape(sign_results)[0] > 100:
                        histology_results, patterns = SpatialDE.aeh.spatial_patterns(X, resid_expr[subset], sign_results, C=n_patterns, l=l, verbosity=1, maxiter= 1000)

                        for topGenes in topGenesArray:
                            print(topGenes)
                            # Select 100 genes that are most likely to fall into each pattern and also select all marker genes:
                            patternGenes = []
                            for pattern in range(n_patterns):
                                rank = ss.rankdata(-1*np.array(histology_results['membership'][histology_results['pattern'] == pattern]))
                                patternGenes.append(np.array(histology_results['g'][histology_results['pattern'] == pattern][[rank[i] <= topGenes or histology_results['g'][histology_results['pattern'] == pattern].iloc[i] in markerGenes for i in range(len(rank))]]))

                            # Further normalize residual expression to express fraction of maximum expression:
                            resid_normed = np.array(resid_expr_subset)
                            genes = resid_expr_subset.columns
                            resid_normed = (resid_normed - resid_normed.min(0)) / resid_normed.ptp(0)

                            colorPalette = np.array(('blue', 'red', 'orange', 'green', 'pink', 'yellow', 'black', 'purple'))

                            # Plot all genes in each pattern and highlight marker genes in colour and with arrow:
                            figsize(5, 2*n_patterns)
                            for i in range(n_patterns):
                                plt.subplot(n_patterns,1, i + 1)
                                count = 0
                                for j in range(len(patternGenes[i])):
                                    colour = 'grey'
                                    alpha = 0.25
                                    label = None 
                                    width = 1
                                    if patternGenes[i][j] in markerGenes:
                                        colour = colorPalette[count]
                                        alpha = 1
                                        label = patternGenes[i][j]
                                        width = 3
                                        count += 1

                                    plt.plot(np.sort(sample_info['x'][subset]), np.array(resid_normed.T[genes == patternGenes[i][j],:])[:,np.argsort(sample_info['x'][subset])][0,:], c=colour, alpha = alpha, label = label, linewidth = width);
                                    plt.title('Pattern {} - {} genes'.format(i, histology_results.query('pattern == @i').shape[0]))
                                    if i == (n_patterns - 1):
                                        plt.xlabel('Relative Cortical Depth')
                                    plt.ylabel('Relative Expression')
                                    plt.legend()
                                    plt.tight_layout()
                            plt.savefig('patternPlots/patterns' + 'Slides' + ''.join(rSlides) + 'NormMethod' + normMethod + 'nGroups' + str(n_patterns) + 'QValue' + str(cutoff) + 'LScale' + str(l) + 'topGenes' + str(topGenes) + '.png') 
                            plt.show()
                            
                            results.to_csv('clusteringResults/' + 'resultList' + 'Slides' + ''.join(rSlides) + 'NormMethod' + normMethod + 'nGroups' + str(n_patterns) + 'QValue' + str(cutoff) + 'LScale' + str(l) + 'topGenes' + str(topGenes) + ".csv")

                        #     for i in histology_results.sort_values('pattern').pattern.unique():
                        #         print('Pattern {}'.format(i))
                        #         print(', '.join(histology_results.query('pattern == @i').sort_values('membership')['g'].tolist()))
                        #         print()