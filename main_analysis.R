### Analysis of Nanostring data:

load('/home/jovyan/data/fetalBrain/Polioudakis/raw_counts_mat.rdata')
raw_counts_mat = as.matrix(raw_counts_mat)

metadata = read.csv('/home/jovyan/data/fetalBrain/Polioudakis/cell_metadata.csv')
clusters = metadata[,'Cluster']

names(clusters) = metadata[,'Cell']
cellStateMatrix = do.call("cbind", tapply(names(clusters), clusters, function(x) rowMeans(raw_counts_mat[,x]))) 

write.csv(cellStateMatrix, '/home/jovyan/data/fetalBrain/Polioudakis/cellStateMatrix.csv')
