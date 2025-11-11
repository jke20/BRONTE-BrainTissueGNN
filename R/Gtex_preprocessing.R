# Author: Jianfeng Ke
# Email: Jianfeng_Ke@student.uml.edu
# Advisor: Rachel Melamed
# Date: 04/23/2023
# Goal: Data preprocessing for Gtex data

## library
setwd("/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen")

# ## Load the data
# AllenMatchDF <- read.csv(paste0(getwd(), "/data/modified_data/AllenMatch.csv"),
#                          row.names = 1, header = T, stringsAsFactors = F)
# ModuleMatch <- read.csv(paste0(getwd(), "/data/modified_data/ModuleMatch.csv"),
#                         row.names = 1, header = T, stringsAsFactors = F)
# GtexMatch <- read.csv(paste0(getwd(), "/data/modified_data/GtexMatch.csv"), 
#                       row.names = 1, header = T, stringsAsFactors = F)

#####-----GtexMatch-----#####
region_match <- c("Amygdala", "Anterior_cingulate_cortex_BA24", "Caudate_basal_ganglia",
                  "Cerebellum", "Cortex", "Hippocampus", "Hypothalamus", 
                  "Nucleus_accumbens_basal_ganglia", "Putamen_basal_ganglia", 
                  "Substantia_nigra")

AMY <- read.table(paste0(getwd(), "/data/gtex_per_tissue/", 
                         region_match[1], ".allen_match.txt.gz"), header = T)
# Add the information of the name of region
AMY <- rbind(c(NA, rep(region_match[1], ncol(AMY)-1)), AMY)
colnames(AMY) <- c("gene_id", paste0(colnames(AMY)[-1],".",1))
AMY[,1] <- c("Region", as.character(AMY[,1])[-1])
GtexMatch <- AMY

## Count the number of subject in each region file
num_subject <- vector(length = 10)
num_subject[1] <- ncol(AMY)-1

for (i in 2:10) {
  DF <- read.table(paste0(getwd(), "/data/gtex_per_tissue/", 
                          region_match[i], ".allen_match.txt.gz"), header = T)
  DF <- rbind(c(NA, rep(region_match[i], ncol(DF)-1)), DF)
  colnames(DF) <- c("gene_id", paste0(colnames(DF)[-1], ".", i))
  DF[,1] <- c("Region", as.character(DF[,1])[-1])
  num_subject[i] <- ncol(DF)-1
  GtexMatch <- merge(GtexMatch, DF, by="gene_id")
}
rm(i, AMY, DF)

# Reformat Gtex dataframe
rid <- which(GtexMatch[,1]=="Region")
GtexMatch <- rbind(GtexMatch[rid,], GtexMatch[-rid,])
GtexMatch <- rbind(colnames(GtexMatch), GtexMatch)
rownames(GtexMatch) <- GtexMatch[,1]
rownames(GtexMatch)[1:2] <- c("ID", "regions")
for (i in 2:ncol(GtexMatch)) {
  string <- strsplit(GtexMatch[1,i], "[.]")[[1]]
  GtexMatch[1,i] <- paste0(string[1],".",string[2])
}
GtexMatch <- GtexMatch[,-1]
rm(i, rid, string)

# Write csv files
write.csv(GtexMatch, paste0(getwd(), "/data/modified_data/GtexMatch.csv"), 
          col.names = T, row.names = T)


#####-----AllenMatchDF-----#####
AllenMatch <- read.table(paste0(getwd(),"/data/gtex_allen_aligned/allen_gtex_match.txt"), 
                         header=T, sep="\t", row.names = 1, stringsAsFactors = F)

# Copy the AllenMatch master file and rename cols and rows
AllenMatchDF <- matrix(NA, nrow=nrow(AllenMatch)+1, ncol=ncol(AllenMatch))
AllenMatchDF <- as.data.frame(AllenMatchDF)
AllenMatchDF[1,] <- colnames(AllenMatch)
AllenMatchDF[2:nrow(AllenMatchDF),] <- AllenMatch
rownames(AllenMatchDF)[1:2] <- c("ID", "regions")
rownames(AllenMatchDF)[3:nrow(AllenMatchDF)] <- rownames(AllenMatch)[2:nrow(AllenMatch)]
colnames(AllenMatchDF) <- 1:ncol(AllenMatchDF)
# Sub-string the IDs
for (i in 1:length(colnames(AllenMatchDF))) {
  string <- strsplit(AllenMatchDF[1,i], "[.]")[[1]]
  AllenMatchDF[1,i] <- string[1]
}

# Remove master files
rm(AllenMatch, i, string)

write.csv(AllenMatchDF, paste0(getwd(), "/data/modified_data/AllenMatch.csv"), 
          col.names = T, row.names = T)


#####-----AllenMatchDF-----#####
AllenExpr <- read.table(paste0(getwd(),"/data/gtex_allen_aligned/gtex_expr.txt"), header = T, 
                        sep="\t", row.names = 1, stringsAsFactors = F)
# Copy the AllenExpr master file and rename cols and rows
AllenExprDF <- AllenExpr
rownames(AllenExprDF)[1:2] <- c("ID", "regions")
AllenExprDF[2,] <- AllenExprDF[1,]
AllenExprDF[1,] <- colnames(AllenExpr)
colnames(AllenExprDF) <- 1:ncol(AllenExprDF)
# Sub-string the IDs
for (i in 1:length(colnames(AllenExprDF))) {
  string <- strsplit(AllenExprDF[1,i], "[.]")[[1]]
  AllenExprDF[1,i] <- paste0(string[1], ".", string[2])
}

write.csv(AllenExprDF, paste0(getwd(), "/data/modified_data/AllenExprDF.csv"), 
          col.names = T, row.names = T)


#####-----ModuleMatch-----#####
ModuleMatch <- read.table(paste0(getwd(),"/data/gtex_allen_aligned/modules_matched.txt"), 
                          header=T, sep="\t", row.names = 1, stringsAsFactors = F)
write.csv(ModuleMatch, paste0(getwd(), "/data/modified_data/ModuleMatch.csv"), 
          col.names = T, row.names = T)





