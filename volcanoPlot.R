# Analysis of Nanostring data:

library(EnhancedVolcano)
library('DESeq2')

# Load markergenes

markers = read.csv('/home/jovyan/data/fetalBrain/Polioudakis/clusterMarkers.csv')

EXDP1_markers = markers[markers[,'cluster'] == 'ExDp1' & markers[,'p_val_adj'] == 0,'X']

vRG_markers = markers[markers[,'cluster'] == 'oRG' & markers[,'p_val_adj'] == 0,'X'][1:10]
IP_markers = markers[markers[,'cluster'] == 'IP' & markers[,'p_val_adj'] == 0,'X'][1:10]

vRG_markers = unlist(lapply(1:length(vRG_markers), function(x) strsplit(as.character(vRG_markers[x]), split = '\\.')[[1]][1]))
IP_markers = unlist(lapply(1:length(IP_markers), function(x) strsplit(as.character(IP_markers[x]), split = '\\.')[[1]][1]))

vRG_markers = c(vRG_markers, 'HOPX')
IP_markers = c(IP_markers)

metadata = read.csv('/home/jovyan/KR_NAS/NanoString sequencing all annotations 2020.02.10.csv')
counts = read.delim('/home/jovyan/KR_NAS/Sanger_288ROIs_TargetCountMatrix.txt', row.names = 1)
genes = rownames(counts)
counts = counts[,match(metadata$Sanger_sampleID[metadata$Sanger_sampleID %in% colnames(counts)], colnames(counts),)]

subset_EOMESpos = metadata$age == '19pcw' & metadata$AOI_type == 'EOMESpos' & !is.na(metadata$Radial_position) & metadata$slide == '00MU'
subset_HOPXpos = metadata$age == '19pcw' & metadata$AOI_type == 'HOPXpos' & !is.na(metadata$Radial_position) & metadata$slide == '00MU'

counts_subset = as.matrix(cbind(counts[, subset_EOMESpos], counts[, subset_HOPXpos]))

# rowCounts = rowSums(counts_subset) 
# counts_subset = counts_subset[rowCounts > 100, ]
# genes = genes[rowCounts > 100]

metadata_subset = rbind(metadata[subset_EOMESpos,], metadata[subset_HOPXpos,])
metadata_subset$AOI_type = factor(metadata_subset$AOI_type)
counts_subset = apply(counts_subset, 2, as.integer)

dds <- DESeqDataSetFromMatrix(countData = counts_subset,
                              colData = metadata_subset,
                              design= ~ AOI_type)

dds <- DESeq(dds)
resultsNames(dds) # lists the coefficients
res <- results(dds)

#removeMarkers = c('COL11A1', 'TMSB4X', 'CA12', 'CLU', 'JUN', 'CRYAB', 'SOX4', 'HES1', 'SOX9', 'SPARC', 
                  # 'VIM', 'TFAPC2', 'CDO1', 'NFE2L2', 'GATM', 'TFAP2C', 'ITGB', 'CNN3', 'SOX2', 'PON2',
                  # 'ITGB8', 'GLI13', 'PAX6', 'GLI3', 'RCN1', 'TAGLN2', 'FGFBP3', 'AKAP12', 'GPM6B', 'INTU',
                  # 'HSPA1A', 'PLPP3', 'MT2A', 'HSPA1B', 'PSAT1', 'IQGAP2')
#vRG_markers = vRG_markers[!vRG_markers %in% removeMarkers]
#IP_markers = IP_markers[!IP_markers %in% removeMarkers]


pdf('BeautifulVolcanoVeryTall.pdf', width = 20, height = 10)
EnhancedVolcano(res,
                lab = genes,
                x = 'log2FoldChange',
                y = 'pvalue',
                pCutoff = 0.01,
                FCcutoff = 1,
                labSize = 5,
                selectLab = c(vRG_markers, IP_markers))
dev.off()

write.csv(res,'/home/jovyan/KR_NAS/DEseq_results.csv')



