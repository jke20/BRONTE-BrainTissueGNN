# install the tensorBF R package if needed
# CP_package_path = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/"
# install.packages(paste0("/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/", 
#                         "tensorBF_1.0.2.tar.gz"), repos = NULL, type = "source")
.libPaths("/home/jianfeng_ke_student_uml_edu/R/x86_64-conda-linux-gnu-library/4.1")
library(tensorBF)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Access and use the arguments
pred_subjects_string <- args[1]     # the subject to predict
pred_region <- args[2]      # the region to predict
K <- as.numeric(args[3])    # the number of conponents
prop <- as.numeric(args[4]) # prop in noiseProp, 0.1~0.9
conf <- as.numeric(args[5]) # conf in noiseProp, 0.1~10

# these are the subjects with all 10 regions in gtex
subject_w_allregions <- 
  c("GTEX.1313W", "GTEX.13NYB", "GTEX.13O3O", "GTEX.13O3Q", "GTEX.13OW8", 
    "GTEX.145LS", "GTEX.15DCD", "GTEX.15ER7", "GTEX.17EVP", "GTEX.1A3MX", 
    "GTEX.1B8L1", "GTEX.1E1VI", "GTEX.1EMGI", "GTEX.1F48J", "GTEX.1GMR8", 
    "GTEX.1GN73", "GTEX.1H1ZS", "GTEX.1H3O1", "GTEX.1J1OQ", "GTEX.1J8Q2", 
    "GTEX.N7MT", "GTEX.NPJ8", "GTEX.Q2AG","GTEX.QDT8", "GTEX.YFC4")

# build a vector for predicted subjects
pred_subjects = strsplit(pred_subjects_string, " ")[[1]]

CP_gtex_subject_region <- function(pred_subjects, pred_region, K, prop, conf) {
  
  data_path = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen"
  ## Load the data (overlapped genes in all 10 regions)
  AllenMatchDF <- read.csv(paste0(data_path, "/data/modified_data/AllenMatch.csv"),
                           row.names = 1, header = T, stringsAsFactors = F)
  ModuleMatch <- read.csv(paste0(data_path, "/data/modified_data/ModuleMatch.csv"),
                          row.names = 1, header = T, stringsAsFactors = F)
  GtexMatch <- read.csv(paste0(data_path, "/data/modified_data/GtexMatch.csv"),
                        row.names = 1, header = T, stringsAsFactors = F)
  
  # find the intersected genes between gtex and allen
  gene_w_module = intersect(rownames(GtexMatch), ModuleMatch$Gene)
  # find the genes in Gtex with module info
  GtexMatch_Module = GtexMatch[rownames(GtexMatch) %in% gene_w_module, ]
  GtexMatch_Module = rbind(GtexMatch[1:2, ], GtexMatch_Module)
  
  # find the subjects who have >= 9 regions
  region_counts <- table(unlist(GtexMatch_Module[1,]))
  pick_subjects <- names(region_counts[region_counts >= 9])
  subject_w_allregions <- names(region_counts[region_counts >= 10])
  
  # build a 3-D tensor in R
  region_match <- c("Amygdala", "Anterior_cingulate_cortex_BA24", 
                    "Caudate_basal_ganglia", "Cerebellum", "Cortex", "Hippocampus", 
                    "Hypothalamus", "Nucleus_accumbens_basal_ganglia", 
                    "Putamen_basal_ganglia", "Substantia_nigra")
  
  # 1st dim: subject; 2nd dim: region; 3rd dim: gene
  # build the Y 3-D tensor
  n_subject = length(pick_subjects)
  n_region = length(region_match)
  n_gene = length(gene_w_module)
  Y = array(NA, dim = c(n_subject, n_region, n_gene))
  dimnames(Y) <- list(
    subject = pick_subjects,
    region = region_match,
    gene = gene_w_module
  )
  
  for (i in 1:length(colnames(GtexMatch_Module))) {
    ID = GtexMatch_Module[1,i]
    region = GtexMatch_Module[2,i]
    if (ID %in% pick_subjects) {
      idx1 = which(pick_subjects==ID)
      idx2 = which(region_match==region)
      Y[idx1, idx2, ] = as.numeric(GtexMatch_Module[3:nrow(GtexMatch_Module),i])
    }
  }
  
  # find the targets
  predict_subjects = pred_subjects
  predict_region = pred_region
  K = K
  prop = prop
  conf = conf
  
  # extract the target and set the target to NA
  target = as.vector(Y[predict_subjects, predict_region, ])
  Y[predict_subjects, predict_region, ] <- NA
  
  # run tensorBF
  CP_result <- tensorBF(Y=Y, K=K, noiseProp=c(prop, conf))
  Y_predict = predictTensorBF(Y, CP_result)
  
  # extract the prediction
  predict_region_idx = which(region_match==predict_region)
  prediction = as.vector(Y_predict[predict_subjects, predict_region_idx, ])
  
  # calculate MSE
  mse_error = mean((target - prediction)^2)
  
  # Create a data frame to store the results
  # output_data <- data.frame(target = target, prediction = prediction)
  # Define the output CSV file name
  output_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/CP_results/'
  output_file <- paste0(output_path, pred_subjects_string, "_", predict_region, "_", 
                        K, "_", prop, "_", conf, ".csv")
  # Write the data frame to a CSV file
  # write.csv(output_data, file = output_file)
  print(paste0("Subject: ", pred_subjects_string))
  print(paste0("Region: ", pred_region))
  print(paste0("K: ", K))
  print(paste0("prop: ", prop))
  print(paste0("conf: ", conf))
  print(paste0("MSE: ", mse_error))
}

CP_gtex_subject_region(pred_subjects, pred_region, K, prop, conf)
# Rscript CP_gtex_subject_region_multiple.R 'GTEX.1313W GTEX.13NYB GTEX.13O3O GTEX.13O3Q GTEX.13OW8' "Putamen_basal_ganglia" 10 0.5 0.5

# # mse
# output_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/CP_results/'
# output_file <- paste0(output_path, predict_subject, "_", predict_region, "_", 60, ".csv")
# Y_pred = read.csv(paste0(output_file), row.names = 1)
# squared_diff <- (Y_pred$target - Y_pred$prediction)^2
# mse <- mean(squared_diff)
# print(mse)

# choose K
# subject = 'GTEX.13O3O'
# K_list = c(17,20,25,30,40,50,60,70)
# mse_list = vector(length = length(K_list))
# for (i in 1:length(K_list)) {
#   output_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/CP_results/'
#   output_file <- paste0(output_path, subject, "_", predict_region, "_", K_list[i], ".csv")
#   Y_pred = read.csv(paste0(output_file), row.names = 1)
#   squared_diff <- (Y_pred$target - Y_pred$prediction)^2
#   mse_list[i] <- mean(squared_diff)
# }

# library(ggplot2)
# # Create a data frame from K_list and mse_list
# data <- data.frame(K = K_list, MSE = mse_list)
# # Create the line plot using ggplot2
# ggplot(data, aes(x = K, y = MSE)) +
#   geom_line() +
#   labs(x = "K", y = "MSE", title = "MSE vs. K, subject: GTEX.13O3O")

