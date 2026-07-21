tmp_folder <- "SET_PROJECT_FOLDER_HERE"
setwd(tmp_folder)

library(ggplot2)
library(gamlss)
library(gamlss.dist)

sys.source(
  file.path(tmp_folder, "brainchart_section3_DTI_FA_MD_functions.R"),
  envir = globalenv()
)
load(file.path(tmp_folder, "brainchart_data_section3_DTI_FA_MD.Rdata"))

output_dir <- file.path(tmp_folder, "brainchart_section3_DTI_FA_MD_outputs")
figure_dir <- file.path(output_dir, "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

FA_columns <- c(
  "FA (Mean)" = "fa_mean",
  "FA (Median)" = "fa_median"
)

MD_columns <- c(
  "MD (Mean)" = "md_mean",
  "MD (Median)" = "md_median"
)

FA_fit_results <- lapply(
  FA_columns,
  function(volume_column) {
    fit_dti_legacy_reference_curve(
      tmp_data_all_healthy_FA,
      volume_column,
      reference_site = "baseline",
      mu_df = 20,
      sigma_df = 3
    )
  }
)

MD_fit_results <- lapply(
  MD_columns,
  function(volume_column) {
    fit_dti_legacy_reference_curve(
      tmp_data_all_healthy_MD,
      volume_column,
      reference_site = "baseline",
      mu_df = 20,
      sigma_df = 3
    )
  }
)

FA_breaks <- make_dti_calibration_breaks(tmp_data_all_healthy_FA$LogAge)
MD_breaks <- make_dti_calibration_breaks(tmp_data_all_healthy_MD$LogAge)

DTI_ref_region_calibrated_list_FA <- lapply(
  names(FA_fit_results),
  function(region) {
    smoothed_curve <- smooth_dti_legacy_reference(
      FA_fit_results[[region]]$reference_curve,
      df_smooth = 12,
      smooth_start = log(300 + 60),
      smooth_apply = log(300 + 90),
      blend_width = 0.05
    )

    calibrate_dti_legacy_reference(
      smoothed_curve,
      FA_fit_results[[region]]$observed_data,
      FA_breaks,
      calibration_df = 7
    )
  }
)
names(DTI_ref_region_calibrated_list_FA) <- names(FA_columns)

DTI_ref_region_calibrated_list_MD <- lapply(
  names(MD_fit_results),
  function(region) {
    smoothed_curve <- smooth_dti_legacy_reference(
      MD_fit_results[[region]]$reference_curve,
      df_smooth = 12,
      smooth_start = log(300 + 60),
      smooth_apply = log(300 + 90),
      blend_width = 0.05
    )

    calibrate_dti_legacy_reference(
      smoothed_curve,
      MD_fit_results[[region]]$observed_data,
      MD_breaks,
      calibration_df = 7
    )
  }
)
names(DTI_ref_region_calibrated_list_MD) <- names(MD_columns)

healthy_coverage_99_DTI_FA <- do.call(
  rbind,
  lapply(
    names(FA_columns),
    function(region) {
      coverage <- calculate_dti_coverage_99(
        DTI_ref_region_calibrated_list_FA[[region]],
        FA_fit_results[[region]]$observed_data
      )
      coverage$Region <- region
      coverage[, c("Region", "Sex", "N", "Coverage_99")]
    }
  )
)

healthy_coverage_99_DTI_MD <- do.call(
  rbind,
  lapply(
    names(MD_columns),
    function(region) {
      coverage <- calculate_dti_coverage_99(
        DTI_ref_region_calibrated_list_MD[[region]],
        MD_fit_results[[region]]$observed_data
      )
      coverage$Region <- region
      coverage[, c("Region", "Sex", "N", "Coverage_99")]
    }
  )
)

for (region in names(FA_columns)) {
  plot <- plot_dti_legacy_reference(
    DTI_ref_region_calibrated_list_FA[[region]],
    FA_fit_results[[region]]$observed_data
  )

  ggsave(
    file.path(figure_dir, paste0(gsub("[ ()]", "_", region), "_ref.pdf")),
    plot = plot,
    width = 10,
    height = 5,
    dpi = 300,
    bg = "white"
  )
}

for (region in names(MD_columns)) {
  plot <- plot_dti_legacy_reference(
    DTI_ref_region_calibrated_list_MD[[region]],
    MD_fit_results[[region]]$observed_data
  )

  ggsave(
    file.path(figure_dir, paste0(gsub("[ ()]", "_", region), "_ref.pdf")),
    plot = plot,
    width = 10,
    height = 5,
    dpi = 300,
    bg = "white"
  )
}

save(
  DTI_ref_region_calibrated_list_FA,
  DTI_ref_region_calibrated_list_MD,
  file = file.path(output_dir, "brainchart_results_section3_DTI_FA_MD.Rdata"),
  compress = "xz"
)

write.csv(
  healthy_coverage_99_DTI_FA,
  file.path(output_dir, "healthy_coverage_99_DTI_FA.csv"),
  row.names = FALSE
)

write.csv(
  healthy_coverage_99_DTI_MD,
  file.path(output_dir, "healthy_coverage_99_DTI_MD.csv"),
  row.names = FALSE
)
