
# conda create -n run_WGCNA r-essentials r-base
# conda install -c conda-forge r-RSQLite
# conda update -c conda-forge r-blob
args <- commandArgs(trailingOnly = TRUE)
subject <- args[1] # GTEX-13NYB
print(paste("The GTEx subject is:", subject))

WGCNA_lib_path <- "/home/jianfeng_ke_student_uml_edu/R/x86_64-conda-linux-gnu-library/Run_WGCNA/"
mice_lib_path <- "/home/jianfeng_ke_student_uml_edu/R/x86_64-conda-linux-gnu-library/Run_mice/"
R_path <- "/home/jianfeng_ke_student_uml_edu/R/x86_64-conda-linux-gnu-library/4.2.3/"
# install.packages("BiocManager", lib=WGCNA_lib_path)
# BiocManager::install("WGCNA", lib.loc=WGCNA_lib_path)
.libPaths(c(WGCNA_lib_path,
            mice_lib_path,
            R_path,
            "/usr/local/lib/R/site-library",
            "/usr/local/lib/R/library",
            "/modules/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.4.0/r-4.2.0-3kitpfbxevyhxd2adiznenkjqqdbekzs/rlib/R/library"))
# options(repos = c(CRAN = "https://cloud.r-project.org/"))
# install.packages("BiocManager")
# install.packages("Hmisc")
# BiocManager::install("WGCNA")
library(WGCNA)
library(dplyr)

# Load gtex data
gtex_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/12052023/'
gt = read.csv(paste0(gtex_path, "new_normed_gtex_gtex_allen_gene.txt"), 
              header = TRUE, sep = "\t", row.names = 1)

# load the GO model prediction for each subject
go_prediction_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/ATG_8_or_less/Prediction/300_500_by_subject/'
file_name = paste0(subject, "_300_500_trainable_LMfromGTEx.csv")
exp = read.csv(paste0(go_prediction_dir, file_name), header = TRUE, row.names = 1)
colnames(exp) = sub("^X", "", colnames(exp))
exp = t(exp)

plotpower <- function(expr){
  powers = c(c(1:10), seq(from = 12, to=20, by=2))
  # Call the network topology analysis function
  sft = pickSoftThreshold(expr, powerVector = powers, verbose = 5, corOptions=list(method = 'spearman'))
  
  sizeGrWindow(9, 5)
  par(mfrow = c(1,2));
  cex1 = 0.9;
  # Scale-free topology fit index as a function of the soft-thresholding power
  plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
       xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit, signed R^2",type="n",
       main = paste("Scale independence"));
  text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
       labels=powers,cex=cex1,col="red");
  #
  abline(h=0.90,col="red")
  # Mean connectivity as a function of the soft-thresholding power
  plot(sft$fitIndices[,1], sft$fitIndices[,5],
       xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
       main = paste("Mean connectivity"))
  text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")
}

modulemaker <- function(expr, softPower ,cortype="spearman"){
  adjacency = adjacency(expr, power = softPower, corOptions = list(method=cortype))
  # Turn adjacency into topological overlap
  TOM = TOMsimilarity(adjacency);
  dissTOM = 1-TOM
  # Call the hierarchical clustering function
  geneTree = hclust(as.dist(dissTOM), method = "average");
  # Plot the resulting clustering tree (dendrogram)
  # sizeGrWindow(12,9)
  # plot(geneTree, xlab="", sub="", main = "Drug clustering on TOM-based dissimilarity",
  #      labels = FALSE, hang = 0.04);
  
  # Set minimum module size
  minModuleSize = 30;
  # Module identification using dynamic tree cut:
  dynamicMods = cutreeDynamic(dendro = geneTree, distM = dissTOM, cutHeight=0.999,
                              deepSplit = 4, pamRespectsDendro = FALSE,
                              minClusterSize = minModuleSize);
  table(dynamicMods)
  
  dynamicColors = labels2colors(dynamicMods)
  table(dynamicColors)
  # Plot the dendrogram and colors underneath
  sizeGrWindow(8,6)
  plotDendroAndColors(geneTree, dynamicColors, "Dynamic Tree Cut",
                      dendroLabels = FALSE, hang = 0.03,
                      addGuide = TRUE, guideHang = 0.05,
                      main = "Drug dendrogram and module colors")
  
  return(list(dynamicMods, dynamicColors))
}

softPower = 14
# create modules
module_results <- modulemaker(exp, softPower)
# store the modules and colors
dynamicMods <- module_results[[1]]
dynamicColors <- module_results[[2]]
modules = data.frame(Gene = colnames(exp), Cluster = dynamicColors)
modules <- modules %>%
  mutate(Module = paste0("M", as.numeric(factor(Cluster))))
# length(unique(modules$Cluster))
# table(modules$Cluster
save_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/WGCNA/raw_modules/'
file_name = paste0(subject, '_raw_modules.csv')
write.csv(modules, paste0(save_dir, file_name), row.names = FALSE)

