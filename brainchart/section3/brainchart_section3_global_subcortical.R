tmp_folder <- "SET_PROJECT_FOLDER_HERE"
setwd(tmp_folder)

library(ggplot2)
library(gamlss)
library(gamlss.dist)

sys.source(
  file.path(tmp_folder, "brainchart_section3_global_subcortical_functions.R"),
  envir = globalenv()
)
load(file.path(tmp_folder, "brainchart_data_section3_global_subcortical.Rdata"))

tmp_output_dir <- file.path(
  tmp_folder,
  "brainchart_section3_global_subcortical_outputs"
)
tmp_figure_dir <- file.path(tmp_output_dir, "figures")
dir.create(tmp_figure_dir, showWarnings = FALSE, recursive = TRUE)

global_regions <- c(
  "GMV",
  "WMV",
  "Ventricles",
  "CBM",
  "BS",
  "Subcortical"
)

global_region_names <- c(
  GMV = "GMV",
  WMV = "WMV",
  Ventricles = "Ventricles",
  CBM = "CBM",
  BS = "BS",
  Subcortical = "Subcortical"
)

global_columns <- list(
  GMV = "GM",
  WMV = "WM",
  Ventricles = "Ventricles",
  CBM = "CBM",
  BS = "BS",
  Subcortical = "Subcortical"
)

subcortical_regions <- c(
  "HippocampusTransformed",
  "AmygdalaTransformed",
  "CaudateTransformed",
  "PutamenTransformed",
  "PallidumTransformed",
  "Accumbens.areaTransformed",
  "Thalamus.ProperTransformed",
  "VentralDCTransformed"
)

subcortical_region_names <- c(
  HippocampusTransformed = "Hippocampus",
  AmygdalaTransformed = "Amygdala",
  CaudateTransformed = "Caudate",
  PutamenTransformed = "Putamen",
  PallidumTransformed = "Pallidum",
  Accumbens.areaTransformed = "Accumbens-area",
  Thalamus.ProperTransformed = "Thalamus-Proper",
  VentralDCTransformed = "Ventral-DC"
)

subcortical_columns <- list(
  HippocampusTransformed = c("L_Hippocampus", "R_Hippocampus"),
  AmygdalaTransformed = c("L_Amygdala", "R_Amygdala"),
  CaudateTransformed = c("L_Caudate", "R_Caudate"),
  PutamenTransformed = c("L_Putamen", "R_Putamen"),
  PallidumTransformed = c("L_Pallidum", "R_Pallidum"),
  Accumbens.areaTransformed = c("L_Accumbens", "R_Accumbens"),
  Thalamus.ProperTransformed = c("L_Thalamus", "R_Thalamus"),
  VentralDCTransformed = c("L_Ventral_DC", "R_Ventral_DC")
)

subcortical_sum_columns <- c(
  "L_Hippocampus",
  "R_Hippocampus",
  "L_Amygdala",
  "R_Amygdala",
  "L_Caudate",
  "R_Caudate",
  "L_Putamen",
  "R_Putamen",
  "L_Pallidum",
  "R_Pallidum",
  "L_Thalamus",
  "R_Thalamus",
  "L_Accumbens",
  "R_Accumbens"
)
calibration_breaks <- tmptmp_breaks
calibration_df <- 7

tmp_data <- tmp_data_archive_0510
tmp_data$Subcortical <- rowSums(
  tmp_data[, subcortical_sum_columns, drop = FALSE],
  na.rm = TRUE
)

global_model_reference_list <- lapply(
  global_regions[1:3],
  function(region) {
    standardize_reference_curve(
      all_results_global[[region]],
      region = region,
      region_name = global_region_names[[region]]
    )
  }
)
names(global_model_reference_list) <- global_regions[1:3]

CB_model_reference_list <- lapply(
  names(all_results_CB),
  function(region) {
    standardize_reference_curve(
      all_results_CB[[region]],
      region = region,
      region_name = region
    )
  }
)
names(CB_model_reference_list) <- names(all_results_CB)

CBM_reference_curve <- combine_cerebellum_reference(
  CB_model_reference_list[["Cerebellum.CortexTransformed"]],
  CB_model_reference_list[["Cerebellum.White.MatterTransformed"]]
)

global_observed_list <- lapply(
  global_regions,
  function(region) {
    prepare_observed_region(tmp_data, global_columns[[region]])
  }
)
names(global_observed_list) <- global_regions

BS_gamlss_result <- fit_gamlss_reference_curve(
  global_observed_list[["BS"]],
  region = "BS",
  region_name = global_region_names[["BS"]],
  selection_method = "bic"
)

Subcortical_gamlss_result <- fit_gamlss_reference_curve(
  global_observed_list[["Subcortical"]],
  region = "Subcortical",
  region_name = global_region_names[["Subcortical"]],
  selection_method = "bic"
)

global_gamlss_reference_list <- list(
  BS = BS_gamlss_result,
  Subcortical = Subcortical_gamlss_result
)

global_ref_region_uncalibrated_list <- c(
  global_model_reference_list,
  list(CBM = CBM_reference_curve),
  global_gamlss_reference_list
)
global_ref_region_uncalibrated_list <-
  global_ref_region_uncalibrated_list[global_regions]

global_ref_region_calibrated_list <- lapply(
  global_regions,
  function(region) {
    calibrate_region_smooth_0428_update(
      global_ref_region_uncalibrated_list[[region]],
      global_observed_list[[region]],
      calibration_breaks,
      calibration_df
    )
  }
)
names(global_ref_region_calibrated_list) <- global_regions

healthy_coverage_99_global <- do.call(
  rbind,
  lapply(
    global_regions,
    function(region) {
      coverage <- calculate_coverage_99(
        global_ref_region_calibrated_list[[region]],
        global_observed_list[[region]]
      )
      coverage$Region <- region
      coverage[, c("Region", "Sex", "N", "Coverage_99")]
    }
  )
)
rownames(healthy_coverage_99_global) <- NULL

Sub_model_regions <- subcortical_regions[1:5]
Sub_model_reference_list <- lapply(
  Sub_model_regions,
  function(region) {
    standardize_reference_curve(
      all_results_subcortical[[region]],
      region = region,
      region_name = subcortical_region_names[[region]]
    )
  }
)
names(Sub_model_reference_list) <- Sub_model_regions

Sub_observed_list <- lapply(
  subcortical_regions,
  function(region) {
    prepare_observed_region(tmp_data, subcortical_columns[[region]])
  }
)
names(Sub_observed_list) <- subcortical_regions

Accumbens_gamlss_result <- fit_gamlss_reference_curve(
  Sub_observed_list[["Accumbens.areaTransformed"]],
  region = "Accumbens.areaTransformed",
  region_name = subcortical_region_names[["Accumbens.areaTransformed"]],
  selection_method = "fixed",
  npoly_mu = 3,
  npoly_sigma = 1
)

Thalamus_gamlss_result <- fit_gamlss_reference_curve(
  Sub_observed_list[["Thalamus.ProperTransformed"]],
  region = "Thalamus.ProperTransformed",
  region_name = subcortical_region_names[["Thalamus.ProperTransformed"]],
  selection_method = "fixed",
  npoly_mu = 3,
  npoly_sigma = 1
)

VentralDC_gamlss_result <- fit_gamlss_reference_curve(
  Sub_observed_list[["VentralDCTransformed"]],
  region = "VentralDCTransformed",
  region_name = subcortical_region_names[["VentralDCTransformed"]],
  selection_method = "fixed",
  npoly_mu = 2,
  npoly_sigma = 3
)

Sub_gamlss_reference_list <- list(
  Accumbens.areaTransformed = Accumbens_gamlss_result,
  Thalamus.ProperTransformed = Thalamus_gamlss_result,
  VentralDCTransformed = VentralDC_gamlss_result
)

Sub_ref_region_uncalibrated_list <- c(
  Sub_model_reference_list,
  Sub_gamlss_reference_list
)
Sub_ref_region_uncalibrated_list <-
  Sub_ref_region_uncalibrated_list[subcortical_regions]

Sub_ref_region_calibrated_list <- lapply(
  subcortical_regions,
  function(region) {
    calibrate_region_smooth_0428_update(
      Sub_ref_region_uncalibrated_list[[region]],
      Sub_observed_list[[region]],
      calibration_breaks,
      calibration_df
    )
  }
)
names(Sub_ref_region_calibrated_list) <- subcortical_regions

healthy_coverage_99_subcortical <- do.call(
  rbind,
  lapply(
    subcortical_regions,
    function(region) {
      coverage <- calculate_coverage_99(
        Sub_ref_region_calibrated_list[[region]],
        Sub_observed_list[[region]]
      )
      coverage$Region <- region
      coverage[, c("Region", "Sex", "N", "Coverage_99")]
    }
  )
)
rownames(healthy_coverage_99_subcortical) <- NULL

for (region in global_regions) {
  plot <- plot_region_truncated_DTI_0502(
    global_ref_region_calibrated_list[[region]],
    global_observed_list[[region]],
    upper_bound = max(calibration_breaks)
  )

  ggsave(
    file.path(
      tmp_figure_dir,
      paste0(global_region_names[[region]], "_ref.pdf")
    ),
    plot = plot,
    width = 10,
    height = 5,
    dpi = 300,
    bg = "white"
  )
}

for (region in subcortical_regions) {
  plot <- plot_region_truncated_DTI_0502(
    Sub_ref_region_calibrated_list[[region]],
    Sub_observed_list[[region]],
    upper_bound = max(calibration_breaks)
  )

  ggsave(
    file.path(
      tmp_figure_dir,
      paste0("Subcortical_", subcortical_region_names[[region]], "_ref.pdf")
    ),
    plot = plot,
    width = 10,
    height = 5,
    dpi = 300,
    bg = "white"
  )
}

save(
  global_ref_region_calibrated_list,
  Sub_ref_region_calibrated_list,
  file = file.path(
    tmp_output_dir,
    "brainchart_results_section3_global_subcortical.Rdata"
  ),
  compress = "xz"
)

write.csv(
  healthy_coverage_99_global,
  file.path(tmp_output_dir, "healthy_coverage_99_global.csv"),
  row.names = FALSE
)
write.csv(
  healthy_coverage_99_subcortical,
  file.path(tmp_output_dir, "healthy_coverage_99_subcortical.csv"),
  row.names = FALSE
)
