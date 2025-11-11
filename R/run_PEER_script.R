# PEER_lib_path <- "/home/jianfeng_ke_student_uml_edu/R/x86_64-conda-linux-gnu-library/Run_PEER"
# .libPaths(c(PEER_lib_path,
#             "/usr/local/lib/R/site-library",
#             "/usr/local/lib/R/library"))
# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
# Access and use the arguments
region_name  <- args[1]               # region name
n            <- as.numeric(args[2])   # n peer

library(peer, quietly=TRUE)
# read the dataframe
new_normed_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/eqtl_analysis/new_bedgz_file/'
filename = paste0(new_normed_dir,region_name,'_newnormed.expression.bed.gz')
nrows <- as.integer(system(paste0("zcat ", filename, " | wc -l | cut -d' ' -f1 "), intern=TRUE, wait=TRUE))
df <- read.table(filename, sep="\t", nrows=nrows, header=TRUE, check.names=FALSE, comment.char="")
# data cleaning
row.names(df) <- df[, 4]
df <- df[, 5:ncol(df)]
M <- t(as.matrix(df))

# # see how does the old df look like
# old_normed_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/eqtl_analysis/GTEx_Analysis_v8_eQTL_expression_matrices/'
# old_filename = paste0(old_normed_dir,'Brain_Amygdala.v8.normalized_expression.bed.gz')
# nrows <- as.integer(system(paste0("zcat ", old_filename, " | wc -l | cut -d' ' -f1 "), intern=TRUE, wait=TRUE))
# old_df = read.table(old_filename, sep="\t", nrows=nrows, header=TRUE, check.names=FALSE, comment.char="")
# row.names(old_df) <- old_df[, 4]
# old_df <- old_df[, 5:ncol(old_df)]
# M <- t(as.matrix(old_df))

# PEER settings
alphaprior_a = 0.001
alphaprior_b = 0.01
epsprior_a = 0.1
epsprior_b = 10
max_iter = 1000

# run PEER
cat(paste0("PEER: estimating hidden confounders (", n, ")\n"))
model <- PEER()
PEER_setNk(model, n)
PEER_setPhenoMean(model, M)
PEER_setPriorAlpha(model, alphaprior_a, alphaprior_b)
PEER_setPriorEps(model, epsprior_a, epsprior_b)
PEER_setNmax_iterations(model, max_iter)
# invisible(PEER_setNk(model, n))
# invisible(PEER_setPhenoMean(model, M))
# invisible(PEER_setPriorAlpha(model, alphaprior_a, alphaprior_b))
# invisible(PEER_setPriorEps(model, epsprior_a, epsprior_b))
# invisible(PEER_setNmax_iterations(model, max_iter))

has.cov <- FALSE
time <- system.time(PEER_update(model))
X <- PEER_getX(model)  # samples x PEER factors
A <- PEER_getAlpha(model)  # PEER factors x 1
R <- t(PEER_getResiduals(model))  # genes x samples

# add relevant row/column names
if (has.cov) {
    cols <- c(colnames(covar.df), paste0("InferredCov",1:(ncol(X)-dim(covar.df)[2])))
} else {
    cols <- paste0("InferredCov",1:ncol(X))
}
rownames(X) <- rownames(M)
colnames(X) <- cols
rownames(A) <- cols
colnames(A) <- "Alpha"
A <- as.data.frame(A)
A$Relevance <- 1.0 / A$Alpha
rownames(R) <- colnames(M)
colnames(R) <- rownames(M)

# write results
cat("PEER: writing results ... ")
# Construct file paths
output_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/eqtl_analysis/PEER_result/'
prefix = region_name
covariates_file <- file.path(output_dir, paste0(prefix, ".PEER_covariates.txt"))
alpha_file <- file.path(output_dir, paste0(prefix, ".PEER_alpha.txt"))
residuals_file <- file.path(output_dir, paste0(prefix, ".PEER_residuals.txt"))
# Write the matrices to the files
write.table(t(X), covariates_file, sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)
write.table(A, alpha_file, sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)
write.table(R, residuals_file, sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)
cat("done.\n")


# combine_covariates.py ${prefix}.PEER_covariates.txt ${prefix} \
#     --genotype_pcs ${genotype_pcs} \
#     --add_covariates ${add_covariates}

