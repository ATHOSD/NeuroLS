library(ggplot2)
library(dplyr)
library(tidyr)

output_dir <- "E:/test_brainchart"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)


source("NN_brainchart_functions.R")


load("volume_reference_variables.RData")
load("disease_age_range_map.RData")
load("healthy_NDI_reference.RData")
load("example_clinical_data.RData")


## ============================================================
## Plot normative centile trajectories
## ============================================================
for (measure_name in names(volume_ref_list)) {
  
  p <- plot_one_volume_normative_curve(
    data = volume_ref_list[[measure_name]],
    measure_name = measure_name
  )
  
  if (measure_name %in% global_measure_names) {
    measure_group <- "global"
  } else if (measure_name %in% subcortical_measure_names) {
    measure_group <- "subcortical"
  } else {
    stop(paste("Unknown measure:", measure_name))
  }
  
  out_file <- file.path(
    output_dir,
    paste0(measure_group, "_", measure_name, "_ref.png")
  )
  
  ggsave(
    filename = out_file,
    plot = p,
    width = 12,
    height = 5,
    dpi = 300,
    units = "in"
  )
}



## ============================================================
## Specify user data
## ============================================================
## Example: test user's DWS data
user_clinical_data <- example_clincial_data_DWS



## ============================================================
## Plot user's clinical data
## ============================================================
user_clinical_data <- add_derived_volume_variables(user_clinical_data)


for (measure_name in names(volume_ref_list)) {
  
  if (measure_name %in% global_measure_names) {
    measure_group <- "global"
  } else if (measure_name %in% subcortical_measure_names) {
    measure_group <- "subcortical"
  }
  
  p <- plot_one_volume_clinical_overlay(
    ref_data = volume_ref_list[[measure_name]],
    measure_name = measure_name,
    user_clinical_data = user_clinical_data,
    volume_columns = volume_columns,
    disease_age_range_map = disease_age_range_map,
    disease_color = "#44AA99"
  )
  
  safe_measure_name <- gsub("[^A-Za-z0-9_\\-]+", "_", measure_name)
  disease_name <- unique(user_clinical_data$Group)
  
  out_file <- file.path(
    output_dir,
    paste0(
      disease_name,
      "_",
      measure_group,
      "_",
      safe_measure_name,
      ".png"
    )
  )
  
  ggsave(
    filename = out_file,
    plot = p,
    width = 12,
    height = 5,
    dpi = 300,
    units = "in"
  )
}


## ============================================================
## Individual centile scoring
## ============================================================
volume_centile_scores <- compute_volume_centile_scores(
  user_clinical_data = user_clinical_data,
  volume_ref_list = volume_ref_list,
  volume_columns = volume_columns
)

median_centile_table <- summarize_volume_median_centile(
  volume_centile_scores = volume_centile_scores
)


median_centile_table$Measurement <- factor(
  median_centile_table$Measurement,
  levels = names(volume_ref_list)
)

median_centile_table <- median_centile_table[order(median_centile_table$Measurement), ]

median_centile_table$Measurement <- as.character(median_centile_table$Measurement)


print(median_centile_table)


disease_name <- unique(user_clinical_data$Group)

write.csv(
  median_centile_table,
  file = file.path(output_dir, paste0(disease_name, "_median_centile_table.csv")),
  row.names = FALSE
)



## ============================================================
## Normative deviation index
## ============================================================
## Global NDI score
user_NDI_global <- compute_user_volume_NDI(
  volume_centile_scores = volume_centile_scores,
  measure_names = global_measure_names
)


## Subcortical NDI score
user_NDI_subcortical <- compute_user_volume_NDI(
  volume_centile_scores = volume_centile_scores,
  measure_names = subcortical_measure_names[1:7]
)


global_violin_out <- plot_user_NDI_violin(
  user_NDI = user_NDI_global,
  healthy_NDI = healthy_NDI_global
)

subcortical_violin_out <- plot_user_NDI_violin(
  user_NDI = user_NDI_subcortical,
  healthy_NDI = healthy_NDI_subcortical
)


write.csv(
  user_NDI_global,
  file = file.path(output_dir, paste0(disease_name, "_global_NDI.csv")),
  row.names = FALSE
)

write.csv(
  user_NDI_subcortical,
  file = file.path(output_dir, paste0(disease_name, "_subcortical_NDI.csv")),
  row.names = FALSE
)

ggplot2::ggsave(
  filename = file.path(output_dir, paste0(disease_name, "_global_NDI_violin.png")),
  plot = global_violin_out$plot,
  width = 5.8,
  height = 5.8,
  units = "in",
  dpi = 300,
  bg = "white"
)

ggplot2::ggsave(
  filename = file.path(output_dir, paste0(disease_name, "_subcortical_NDI_violin.png")),
  plot = subcortical_violin_out$plot,
  width = 5.8,
  height = 5.8,
  units = "in",
  dpi = 300,
  bg = "white"
)





