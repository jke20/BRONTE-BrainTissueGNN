# install packages
.libPaths(c("/home/jianfeng_ke_student_uml_edu/R/x86_64-conda-linux-gnu-library/color+MR/", 
            "/home/jianfeng_ke_student_uml_edu/R/x86_64-conda-linux-gnu-library/Run_mice", 
            .libPaths()))

# Libraries
library(rtracklayer)
library(data.table)
library(dplyr)
library(coloc)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
# Access and use the arguments
allen_region <- args[1]     # the subject to predict
ieu_id <- args[2]           # the region to predict
# allen_region = 4251

# it's better to have at least 256GiB memory to run this script
# check if we already did the mapping already
save_dir = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/coloc+MR/all_snps_with_hg37/"
filepath = paste0(save_dir, allen_region, "_eqtl_result_with_hg19.txt.gz")
# checking
if (file.exists(filepath)) {
  message(paste0("We already mapped hg38 to hg19 for allen region ", allen_region, ". Skip mapping!"))
} else {
  message("Start mapping hg38 to hg19 for allen region ", allen_region, "!")
  
  #####-----mapping hg38 to hg19-----#####
  save_dir <- "/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/eqtl_analysis/fastQTL_for_allen_regions/fastqtl_results/"
  filename <- paste0(allen_region, ".allpairs.txt.gz")
  # Load data
  # eqtl_result = fread(paste0(save_dir, filename), nrows = 5000)
  eqtl_result = fread(paste0(save_dir, filename))
  # extract chromosome and position of the variants
  positions = eqtl_result %>% dplyr::select(variant_id)
  positions = as.data.frame(stringr::str_split_fixed(positions$variant_id, pattern = "_", n = Inf))
  colnames(positions) = c("chromosome", "position", "ref_allele", "alt_allele", "genome_version")
  # Chromosome column must be in the format "chr#"
  eqtl_result$CHR = positions$chromosome
  eqtl_result$position = positions$position
  
  # Import chain object for liftover
  hg38tohg19_chain = import.chain("/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/coloc+MR/liftover/hg38ToHg19.over.chain")
  # Create GRanges from SNP positions
  granges_data = makeGRangesFromDataFrame(eqtl_result,
                                          keep.extra.columns = TRUE,
                                          seqnames.field = "CHR",
                                          start.field = "position",
                                          end.field = "position",
                                          ignore.strand = TRUE)
  remove(eqtl_result)
  
  # Liftover
  eqtl_hg37 = liftOver(x = granges_data, chain = hg38tohg19_chain)
  remove(granges_data)
  # Transform GRanges object to data.frame
  eqtl_hg37_data_frame = as.data.frame(eqtl_hg37)
  remove(eqtl_hg37)
  eqtl_hg37_data_frame = eqtl_hg37_data_frame %>% dplyr::select(-group, -group_name, chr = seqnames, -width, -strand)
  
  # Write the dataframe
  save_dir = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/coloc+MR/all_snps_with_hg37/"
  filepath = paste0(save_dir, allen_region, "_eqtl_result_with_hg19.txt.gz")
  data.table::fwrite(x = eqtl_hg37_data_frame, file = filepath, sep = "\t", row.names = FALSE, nThread = 4)
  message(paste0("Successfully generating hg38->hg19 mapping info for all snps with eqtl results for allen region ", allen_region, "!"))
}



#####-----find the snps around the target snp-----#####
# define the target dataset and the target snps
dataset = ieu_id
message(paste0("Target dataset: ", dataset))

# load the snps with eqtl results and hg19 info
allen_region = allen_region
message("Reading eqtl results for all snps with hg38->hg19 mapping info...")
save_dir = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/coloc+MR/all_snps_with_hg37/"
filename <- paste0(allen_region, "_eqtl_result_with_hg19.txt.gz")
eqtl_hg19 = fread(paste0(save_dir, filename))
message("Success!")
# chech how many snps have different chr in hg38 and hg19
hg38_positions = eqtl_hg19 %>% dplyr::select(variant_id)
hg38_positions = as.data.frame(stringr::str_split_fixed(hg38_positions$variant_id, pattern = "_", n = Inf))
colnames(hg38_positions) = c("chr", "position", "ref_allele", "alt_allele", "genome_version")
eqtl_hg19$hg38_chr = hg38_positions$chr
eqtl_hg19$ref = hg38_positions$ref_allele
eqtl_hg19$alt = hg38_positions$alt_allele
# eqtl_hg19_mis_chr = eqtl_hg19[which(eqtl_hg19$chr!=eqtl_hg19$hg38_chr),]
# length(unique(eqtl_hg19_mis_chr$gene_id))

# read the mr results 
save_dir = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/R/coloc+MR/MR_results/"
filename <- paste0(allen_region, "_", dataset, "_mr_results_method2_local.csv")
mr_result = fread(paste0(save_dir, filename))
message("Successfully reading mr results!")

# target_var_id = mr_result$variant_id[1]
target_var_id = 'chr1_207577223_T_C_b38'
message(paste0("Target variant: ", target_var_id))

# let's use this variant: chr11_74892136_G_A_b38, rsid: rs4944963 as an example
# another example: chr1_207577223_T_C_b38, rs679515
target_gene = mr_result$gene_id[which(mr_result$variant_id==target_var_id)]
target_rsid = mr_result$rsid[which(mr_result$variant_id==target_var_id)]
target_chr = stringr::str_split(target_var_id, "_")[[1]][1]
target_pos = eqtl_hg19$start[which((eqtl_hg19$variant_id==target_var_id) & 
                                     (eqtl_hg19$gene_id==target_gene))]
window_size = 100000
message(paste0("Pick window size: ", window_size))
# find all the SNPs within a certain window 
window_snps = eqtl_hg19[which((eqtl_hg19$chr==target_chr) & 
                                (eqtl_hg19$gene_id==target_gene) & 
                                (eqtl_hg19$start>(target_pos-window_size)) & 
                                (eqtl_hg19$start<(target_pos+window_size))), ]
window_snps = window_snps %>% rename(eqtl_p=pval_nominal, eqtl_b=slope, eqtl_se=slope_se)
window_snps$chr_num = sub("chr", "", window_snps$chr)
# Step 0: chr need to match in hg38 and hg19
window_snps = window_snps %>% dplyr::filter(chr==hg38_chr)

#####-----query the gwas summary-----#####
# load the gwasvcf
library(gwasvcf)
filepath = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/MR+coloc/gwas_vcf/'
set_plink()
set_bcftools()
# load the vcf file
filename = paste0(dataset, '.vcf.gz')
vcffile <- paste0(filepath, filename)
# load the ld file
eur_ref_filename <- "data_maf0.01_rs_ref.bed"
ldfile <- paste0(filepath, eur_ref_filename) %>% gsub(".bed", "", .)
# do the query
chrompos = paste0(window_snps$chr_num, ":", window_snps$start)
gwas_query <- vcf_to_granges(query_gwas(vcffile, chrompos=chrompos, proxies="no", bfile=ldfile, tag_kb=10000, tag_r2=0.8))
gwas_query_dat <- data.frame(gwas_query) %>% 
  dplyr::select(chr=seqnames, start, end, ref=REF, alt=ALT, 
                gwas_p=LP, gwas_b=ES, gwas_se=SE, rsid=ID, id=id)
gwas_query_dat$gwas_p = 10^(-gwas_query_dat$gwas_p)

# merge eqtl and gwas query
# Step 1: ref and alt allele should also match
eqtl_gwas_snp = left_join(window_snps, gwas_query_dat, by = c("chr_num"="chr", "start", "end", "ref", "alt"))
eqtl_gwas_snp = eqtl_gwas_snp[which(!is.na(eqtl_gwas_snp$gwas_p))]
# check if duplicated?
# sum(duplicated(eqtl_gwas_snp$rsid)), Yes!
# duplicated_snps = unique(eqtl_gwas_snp[which(duplicated(eqtl_gwas_snp$rsid)), ]$rsid)
# duplicated_snps_mat = eqtl_gwas_snp[which(eqtl_gwas_snp$rsid %in% duplicated_snps), ]
eqtl_gwas_snp = eqtl_gwas_snp[, .SD[which.min(gwas_p)], by = rsid]

#####-----run coloc.abf-----#####
# # compute the ld
# library(plinkbinr)
# library(ieugwasr)
# plink = get_plink_exe()
# ld_matrix = ld_matrix_local(variants = eqtl_gwas_snp$rsid,
#                             with_alleles = FALSE,
#                             bfile = ldfile,
#                             plink_bin = plink)
# # only pick the snps having LD info
# eqtl_gwas_snp = eqtl_gwas_snp[which(eqtl_gwas_snp$rsid %in% colnames(ld_matrix)), ]

# Prepare eQTL dataset
eqtl_data <- list(
  pvalues = eqtl_gwas_snp$eqtl_p,
  N = nrow(eqtl_gwas_snp),
  position = eqtl_gwas_snp$start,
  MAF = eqtl_gwas_snp$maf,
  beta = eqtl_gwas_snp$eqtl_b,
  varbeta = (eqtl_gwas_snp$eqtl_se)^2, 
  snp = eqtl_gwas_snp$rsid, 
  # LD = ld_matrix, 
  type = "quant"
)

# Prepare GWAS dataset
gwas_data <- list(
  pvalues = eqtl_gwas_snp$gwas_p,
  N = nrow(eqtl_gwas_snp),
  position = eqtl_gwas_snp$start,
  MAF = eqtl_gwas_snp$maf,
  beta = eqtl_gwas_snp$gwas_b,
  varbeta = (eqtl_gwas_snp$gwas_se)^2, 
  snp = eqtl_gwas_snp$rsid, 
  # LD = ld_matrix, 
  type = "quant"
)

# # plot the snps around with gwas p-value
# plot_dataset(gwas_data)
# points(eqtl_gwas_snp$start[which(eqtl_gwas_snp$variant_id==target_var_id)], 
#        -log10(eqtl_gwas_snp$gwas_p[which(eqtl_gwas_snp$variant_id==target_var_id)]), 
#        col = "red", pch = 16, cex = 1.5)

# run coloc.abf()
coloc_result <- coloc.abf(eqtl_data, gwas_data, p1=1e-4, p2=1e-4, p12=1e-5)
# coloc_result <- coloc.abf(eqtl_data, gwas_data, p1=1/(nrow(eqtl_gwas_snp) + 1), p2=1/(nrow(eqtl_gwas_snp) + 1), p12=1/10)





