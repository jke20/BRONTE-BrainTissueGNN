# install.packages("mice", lib = new_lib_path, dependencies = TRUE)
new_lib_path <- "/home/jianfeng_ke_student_uml_edu/R/x86_64-conda-linux-gnu-library/Run_mice"
.libPaths(c(new_lib_path,
            "/usr/local/lib/R/site-library",
            "/usr/local/lib/R/library",
            "/modules/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.4.0/r-4.2.0-3kitpfbxevyhxd2adiznenkjqqdbekzs/rlib/R/library"))
library(mice)
# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
# Access and use the arguments
pred_subject <- args[1]           # the subject to predict, "GTEX-1313W"
pred_region <- args[2]             # the region to predict, "Putamen_basal_ganglia"
missing_N <- as.numeric(args[3])   # the number of missing region is allowed, 0


MICE_gtex_subject_region <- function(pred_subject, pred_region, missing_N) {
  
  # Load gtex data
  gtex_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/12052023/'
  gt = read.csv(paste0(gtex_path, "new_normed_gtex_gtex_allen_gene.txt"), 
                header = TRUE, sep = "\t", row.names = 1)
  allen_gtex_gene = rownames(gt)[3:length(rownames(gt))]
  #allen_gtex_gene = rownames(gt)[3:100]
  
  # find the subjects who have no more than missing_N regions
  region_counts <- table(unlist(gt[2,]))
  pick_subjects <- names(region_counts[region_counts >= (10-missing_N)])
  
  # build a 3-D tensor in R
  region_match <- c("Amygdala", "Anterior_cingulate_cortex_BA24", 
                    "Caudate_basal_ganglia", "Cerebellar_Hemisphere", 
                    "Frontal_Cortex_BA9", "Hippocampus", 
                    "Hypothalamus", "Nucleus_accumbens_basal_ganglia", 
                    "Putamen_basal_ganglia", "Substantia_nigra")
  
  # 1st dim: subject; 2nd dim: region; 3rd dim: gene
  # build the Y 3-D tensor
  n_subject = length(pick_subjects)
  n_region = length(region_match)
  n_gene = length(allen_gtex_gene)
  Y = array(NA, dim = c(n_subject, n_region, n_gene))
  dimnames(Y) <- list(
    subject = pick_subjects,
    region = region_match,
    gene = allen_gtex_gene
  )
  
  for (i in 1:length(colnames(gt))) {
    region = gt[1,i]
    ID = gt[2,i]
    if (ID %in% pick_subjects) {
      idx1 = which(pick_subjects==ID)
      idx2 = which(region_match==region)
      Y[idx1, idx2, ] = as.numeric(as.character(gt[3:(length(allen_gtex_gene)+2),i]))
    }
  }
  
  # Y_mice is all predictions from MICE
  Y_mice = Y
  for (gene in allen_gtex_gene) {
      Y_gene = Y[, , gene]
      Y_gene[pred_subject, pred_region] <- NA
      imputed_data <- mice(Y_gene, seed = 2024)
      completed_data <- complete(imputed_data)
      Y_mice[pred_subject, pred_region, gene] = completed_data[pred_subject, pred_region]
  }
  
  target = Y[pred_subject, pred_region, ]
  prediction = Y_mice[pred_subject, pred_region, ]
  # Create a data frame to store the results
  output_data <- data.frame(target = target, prediction = prediction)
  # Define the output CSV file name
  output_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/Running_MICE/MICE_results/'
  output_file <- paste0(output_path, pred_subject, "_", pred_region, "_", missing_N, ".csv")
  # Write the data frame to a CSV file
  write.csv(output_data, file = output_file)
}

# running MICE
MICE_gtex_subject_region(pred_subject, pred_region, missing_N)


