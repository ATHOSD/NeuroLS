tmp_folder <- "SET_PROJECT_FOLDER_HERE"

sys.source(
  file.path(tmp_folder, "brainchart_section3_milestones_functions.R"),
  envir = globalenv()
)

global_subcortical_data <- new.env()
load(
  file.path(
    tmp_folder,
    "brainchart_data_section3_global_subcortical.Rdata"
  ),
  envir = global_subcortical_data
)

dti_data <- new.env()
load(
  file.path(tmp_folder, "brainchart_data_section3_DTI_FA_MD.Rdata"),
  envir = dti_data
)

global_reference_list <-
  global_subcortical_data$global_ref_region_calibrated_list
subcortical_reference_list <-
  global_subcortical_data$Sub_ref_region_calibrated_list
dti_fa_reference_list <-
  dti_data$DTI_ref_region_calibrated_list_FA
dti_md_reference_list <-
  dti_data$DTI_ref_region_calibrated_list_MD

names(global_reference_list) <- c(
  "GMV",
  "WMV",
  "Ventricles",
  "CBM",
  "BS",
  "Subcortical"
)
global_region_labels <- c(
  GMV = "Gray Matter",
  WMV = "White Matter",
  Ventricles = "Ventricles",
  CBM = "Cerebellum",
  BS = "Brain Stem",
  Subcortical = "Subcortical"
)

names(subcortical_reference_list) <- c(
  "HippocampusTransformed",
  "AmygdalaTransformed",
  "CaudateTransformed",
  "PutamenTransformed",
  "PallidumTransformed",
  "Accumbens.areaTransformed",
  "Thalamus.ProperTransformed",
  "VentralDCTransformed"
)
subcortical_region_labels <- c(
  HippocampusTransformed = "Hippocampus",
  AmygdalaTransformed = "Amygdala",
  CaudateTransformed = "Caudate",
  PutamenTransformed = "Putamen",
  PallidumTransformed = "Pallidum",
  Accumbens.areaTransformed = "Accumbens",
  Thalamus.ProperTransformed = "Thalamus",
  VentralDCTransformed = "Ventral DC"
)

dti_reference_list <- list(
  Mean_FA = dti_fa_reference_list[[1]],
  Median_FA = dti_fa_reference_list[[2]],
  Mean_MD = dti_md_reference_list[[1]],
  Median_MD = dti_md_reference_list[[2]]
)
dti_region_labels <- c(
  Mean_FA = "Mean FA",
  Median_FA = "Median FA",
  Mean_MD = "Mean MD",
  Median_MD = "Median MD"
)

upper_bound <- max(global_subcortical_data$tmptmp_breaks)

global_milestones <- summarize_reference_milestones(
  global_reference_list,
  "Global",
  global_region_labels,
  upper_bound
)

subcortical_milestones <- summarize_reference_milestones(
  subcortical_reference_list,
  "Subcortical",
  subcortical_region_labels,
  upper_bound
)

dti_milestones <- summarize_reference_milestones(
  dti_reference_list,
  "DTI",
  dti_region_labels,
  upper_bound
)

brainchart_section3_milestones <- rbind(
  global_milestones,
  subcortical_milestones,
  dti_milestones
)
rownames(brainchart_section3_milestones) <- NULL

tmp_output_dir <- file.path(
  tmp_folder,
  "brainchart_section3_milestones_outputs"
)
dir.create(tmp_output_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(
  brainchart_section3_milestones,
  file.path(tmp_output_dir, "brainchart_section3_milestones.csv"),
  row.names = FALSE
)
