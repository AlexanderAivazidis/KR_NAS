require(DESeq2)
library(Rcpp)

counts = read.csv('/nfs/team283/aa16/KR_NAS/data/FetalBrain_AllData_AnnData_1_X.csv', row.names = 1)
obs = read.csv('/nfs/team283/aa16/KR_NAS/data/FetalBrain_AllData_AnnData_1_obs.csv')
var = read.csv('/nfs/team283/aa16/KR_NAS/data/FetalBrain_AllData_AnnData_1_var.csv')
colnames(counts) = var[,'SYMBOL']

counts_subset = counts[obs['Pool'] == 'CRcells',]
obs_subset = obs[obs['Pool'] == 'CRcells',]

dds <- DESeqDataSetFromMatrix(countData = t(counts_subset),
                              colData = obs_subset,
                              design= ~ Donor + Tissue)

dds <- DESeq(dds)
resultsNames(dds) # lists the coefficients
res <- results(dds, name="Tissue_Foetal.brain..occipital.lobe._vs_Foetal.brain..frontal.lobe.")
# or to shrink log fold changes association with condition:
res <- lfcShrink(dds, coef="Tissue_Foetal.brain..occipital.lobe._vs_Foetal.brain..frontal.lobe.", type="apeglm")

res = res[order(res[,"log2FoldChange"]),]
res = res[!is.na(res[,'padj']),]
res_subset = res[res[,'padj'] < 0.1,]
res_subset = res_subset[abs(res_subset[,'log2FoldChange']) > 0.5,]
res_subset = res_subset[res_subset[,'baseMean'] > 7.5,]
sort(rownames(res_subset))
dim(res_subset)
