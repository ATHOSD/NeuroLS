## ============================================================
## Fixed sex colors
## ============================================================

volume_sex_colors <- c(
  "F" = "#8B0000",
  "M" = "#00008B",
  "Female" = "#8B0000",
  "Male" = "#00008B"
)


## ============================================================
## Age axis helper
## ============================================================

get_volume_age_axis <- function() {
  ages <- data.frame(
    days = c(
      147,
      280,
      280 + 90,
      280 + 365,
      280 + 1095,
      280 + 1825,
      280 + 3650,
      280 + 7300,
      280 + 14600,
      280 + 29200
    ),
    label = c(
      "21GA",
      "birth",
      "3mo",
      "1y",
      "3y",
      "5y",
      "10y",
      "20y",
      "40y",
      "80y"
    )
  )
  
  list(
    breaks = log(ages$days),
    labels = ages$label
  )
}


## ============================================================
## Centile column helper
## ============================================================

get_volume_centile_columns <- function(data) {
  
  candidates <- list(
    c005 = c("Centile_005"),
    c025 = c("Centile_025"),
    c50  = c("Centile_500", "Centile_50"),
    c975 = c("Centile_975"),
    c995 = c("Centile_995")
  )
  
  out <- lapply(candidates, function(x) {
    tmp <- x[x %in% names(data)][1]
    if (is.na(tmp)) {
      return(NA_character_)
    } else {
      return(tmp)
    }
  })
  
  missing_cols <- names(out)[is.na(unlist(out))]
  
  if (length(missing_cols) > 0) {
    stop(
      paste(
        "Cannot find required centile columns:",
        paste(missing_cols, collapse = ", ")
      )
    )
  }
  
  out
}


## ============================================================
## Y-axis scale helper
## ============================================================

get_volume_y_scale <- function(measure_name) {
  
  if (measure_name == "Accumbens") {
    out <- list(
      scale = 1e3,
      label = "\u00d710^3"
    )
  } else {
    out <- list(
      scale = 1e4,
      label = "\u00d710^4"
    )
  }
  
  out
}


## ============================================================
## Y-axis limit helper
## ============================================================

get_volume_y_limits <- function(data, x_limits, padding = 0.15) {
  
  data_use <- data[
    data$LogAge >= x_limits[1] &
      data$LogAge <= x_limits[2],
  ]
  
  centile_cols <- grep("^Centile_", names(data_use), value = TRUE)
  
  y_values <- unlist(data_use[, centile_cols, drop = FALSE])
  y_values <- y_values[is.finite(y_values)]
  
  y_min <- min(y_values, na.rm = TRUE)
  y_max <- max(y_values, na.rm = TRUE)
  y_range <- y_max - y_min
  
  c(
    y_min - padding * y_range,
    y_max + padding * y_range
  )
}


## ============================================================
## Plot one measure
## ============================================================

plot_one_volume_normative_curve <- function(
    data,
    measure_name,
    y_limits = NULL,
    x_limits = c(log(147), log(280 + 85 * 365)),
    y_padding = 0.15
) {
  
  centile_cols <- get_volume_centile_columns(data)
  age_axis <- get_volume_age_axis()
  y_scale <- get_volume_y_scale(measure_name)
  
  data <- data[
    data$LogAge >= x_limits[1] &
      data$LogAge <= x_limits[2],
  ]
  
  if (is.null(y_limits)) {
    y_limits <- get_volume_y_limits(
      data = data,
      x_limits = x_limits,
      padding = y_padding
    )
  }
  
  p <- ggplot2::ggplot(data, ggplot2::aes(x = LogAge)) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c005]], color = Sex),
      linetype = "dotted",
      linewidth = 1.2,
      alpha = 0.7
    ) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c025]], color = Sex),
      linetype = "dashed",
      linewidth = 1.3,
      alpha = 0.7
    ) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c50]], color = Sex),
      linetype = "solid",
      linewidth = 1.5
    ) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c975]], color = Sex),
      linetype = "dashed",
      linewidth = 1.3,
      alpha = 0.7
    ) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c995]], color = Sex),
      linetype = "dotted",
      linewidth = 1.2,
      alpha = 0.7
    ) +
    
    ggplot2::geom_vline(
      xintercept = log(280),
      linetype = "dotted",
      color = "darkgray",
      linewidth = 1.5
    ) +
    
    ggplot2::facet_wrap(
      ~ Sex,
      ncol = 2,
      axes = "all_y",
      axis.labels = "all_y"
    ) +
    
    ggplot2::scale_color_manual(values = volume_sex_colors) +
    
    ggplot2::scale_x_continuous(
      breaks = age_axis$breaks,
      labels = age_axis$labels,
      limits = x_limits,
      expand = c(0, 0),
      name = NULL
    ) +
    
    ggplot2::scale_y_continuous(
      labels = function(x) x / y_scale$scale,
      expand = ggplot2::expansion(mult = c(0, 0.03)),
      name = NULL
    ) +
    
    ggplot2::annotate(
      "text",
      x = x_limits[1],
      y = Inf,
      label = y_scale$label,
      hjust = -0.1,
      vjust = 1.2,
      size = 8
    ) +
    
    ggplot2::coord_cartesian(ylim = y_limits) +
    
    ggplot2::theme_classic() +
    ggplot2::theme(
      strip.text = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(
        angle = 45,
        hjust = 1,
        size = 17,
        color = "black"
      ),
      axis.text.y = ggplot2::element_text(
        size = 22,
        color = "black"
      ),
      axis.title = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(
        linewidth = 1.5,
        color = "black"
      ),
      axis.ticks = ggplot2::element_line(
        linewidth = 1.5,
        color = "black"
      ),
      axis.ticks.length = grid::unit(0.28, "cm"),
      legend.position = "none",
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      strip.background = ggplot2::element_blank(),
      plot.title = ggplot2::element_blank(),
      plot.subtitle = ggplot2::element_blank(),
      panel.spacing.x = grid::unit(0.6, "cm"),
      aspect.ratio = 1
    )
  
  g <- ggplot2::ggplotGrob(p)
  
  id <- grep("^axis-l", g$layout$name)
  g$widths[unique(g$layout$l[id])] <- grid::unit(1.8, "cm")
  
  g
}



## ============================================================
## Derived volume variables helper
## ============================================================

add_derived_volume_variables <- function(data) {
  
  data %>%
    dplyr::mutate(
      LogAge = log(PostConceptionDays),
      CBM = CBM_Cortex + CBM_WM,
      Subcortical =
        L_Hippocampus + R_Hippocampus +
        L_Amygdala + R_Amygdala +
        L_Caudate + R_Caudate +
        L_Putamen + R_Putamen +
        L_Pallidum + R_Pallidum +
        L_Thalamus + R_Thalamus +
        L_Accumbens + R_Accumbens
    )
}


## ============================================================
## Standardize sex labels
## ============================================================

standardize_volume_sex <- function(data) {
  
  if (!"Sex" %in% names(data)) {
    return(data)
  }
  
  data %>%
    dplyr::mutate(
      Sex = dplyr::case_when(
        Sex %in% c("F", "Female") ~ "Female",
        Sex %in% c("M", "Male") ~ "Male",
        TRUE ~ as.character(Sex)
      )
    )
}


## ============================================================
## Get x-axis information for one clinical group
## ============================================================

get_volume_x_info_for_group <- function(group_name, disease_age_range_map) {
  
  group_name <- unique(as.character(group_name))
  group_name <- group_name[!is.na(group_name)]
  
  if (length(group_name) != 1) {
    stop("user_clinical_data should contain exactly one disease group.")
  }
  
  if (!group_name %in% names(disease_age_range_map)) {
    stop(paste("No age-range setting found for group:", group_name))
  }
  
  disease_age_range_map[[group_name]]
}


## ============================================================
## Prepare clinical data for one volume measure
## ============================================================

prepare_volume_clinical_measure_data <- function(
    user_clinical_data,
    measure_name,
    volume_columns
) {
  
  if (!"Group" %in% names(user_clinical_data)) {
    stop("user_clinical_data must contain a Group column.")
  }
  
  if (!"LogAge" %in% names(user_clinical_data)) {
    stop("user_clinical_data must contain LogAge. Run add_derived_volume_variables() first.")
  }
  
  group_name <- unique(as.character(user_clinical_data$Group))
  group_name <- group_name[!is.na(group_name)]
  
  if (length(group_name) != 1) {
    stop("user_clinical_data should contain exactly one disease group.")
  }
  
  if (!measure_name %in% names(volume_columns)) {
    stop(paste("Cannot find measure in volume_columns:", measure_name))
  }
  
  measure_cols <- volume_columns[[measure_name]]
  
  missing_cols <- measure_cols[!measure_cols %in% names(user_clinical_data)]
  
  if (length(missing_cols) > 0) {
    stop(
      paste(
        "Missing required clinical data columns:",
        paste(missing_cols, collapse = ", ")
      )
    )
  }
  
  data_out <- user_clinical_data
  
  if (length(measure_cols) == 1) {
    data_out$Volume <- data_out[[measure_cols]]
  } else if (length(measure_cols) == 2) {
    data_out$Volume <- rowMeans(
      data_out[, measure_cols, drop = FALSE],
      na.rm = TRUE
    )
  } else {
    stop("volume_columns should contain either one column or left/right columns.")
  }
  
  if (group_name == "PWMI" &&
      (!"Sex" %in% names(data_out) || all(is.na(data_out$Sex)))) {
    
    data_female <- data_out
    data_male <- data_out
    
    data_female$Sex <- "Female"
    data_male$Sex <- "Male"
    
    data_out <- dplyr::bind_rows(data_female, data_male)
    
  } else {
    
    data_out <- standardize_volume_sex(data_out)
    
    if (!"Sex" %in% names(data_out)) {
      stop("user_clinical_data must contain Sex unless Group is PWMI.")
    }
  }
  
  data_out %>%
    dplyr::filter(Sex %in% c("Female", "Male"))
}


## ============================================================
## Y-axis limits for clinical overlay
## ============================================================

get_volume_overlay_y_limits <- function(
    ref_data,
    clinical_measure_data,
    x_info,
    centile_cols,
    padding = 0.15
) {
  
  ref_use <- ref_data %>%
    standardize_volume_sex() %>%
    dplyr::filter(
      LogAge >= x_info$x_min,
      LogAge <= x_info$x_max,
      Sex %in% c("Female", "Male")
    )
  
  y_ref <- c(
    ref_use[[centile_cols$c005]],
    ref_use[[centile_cols$c50]],
    ref_use[[centile_cols$c995]]
  )
  
  clinical_use <- clinical_measure_data %>%
    dplyr::filter(
      LogAge >= x_info$x_min,
      LogAge <= x_info$x_max
    )
  
  y_dat <- clinical_use$Volume
  y_dat <- y_dat[is.finite(y_dat)]
  
  if (length(y_dat) > 10) {
    q_low <- stats::quantile(y_dat, 0.01, na.rm = TRUE)
    q_high <- stats::quantile(y_dat, 0.99, na.rm = TRUE)
    y_dat <- y_dat[y_dat >= q_low & y_dat <= q_high]
  }
  
  y_all <- c(y_ref, y_dat)
  y_all <- y_all[is.finite(y_all)]
  
  if (length(y_all) == 0) {
    return(c(0, 1))
  }
  
  y_min <- min(y_all, na.rm = TRUE)
  y_max <- max(y_all, na.rm = TRUE)
  y_range <- y_max - y_min
  
  if (!is.finite(y_range) || y_range == 0) {
    y_range <- max(abs(y_max), 1) * 0.1
  }
  
  c(
    max(y_min - padding * y_range, 0),
    y_max + padding * y_range
  )
}


## ============================================================
## Plot one measure with clinical overlay
## ============================================================

plot_one_volume_clinical_overlay <- function(
    ref_data,
    measure_name,
    user_clinical_data,
    volume_columns,
    disease_age_range_map,
    disease_color = "#44AA99",
    point_size = 2.0,
    point_alpha = 0.8
) {
  
  group_name <- unique(as.character(user_clinical_data$Group))
  group_name <- group_name[!is.na(group_name)]
  
  x_info <- get_volume_x_info_for_group(
    group_name = group_name,
    disease_age_range_map = disease_age_range_map
  )
  
  centile_cols <- get_volume_centile_columns(ref_data)
  y_scale <- get_volume_y_scale(measure_name)
  
  ref_plot <- ref_data %>%
    standardize_volume_sex() %>%
    dplyr::filter(
      LogAge >= x_info$x_min,
      LogAge <= x_info$x_max,
      Sex %in% c("Female", "Male")
    )
  
  clinical_plot <- prepare_volume_clinical_measure_data(
    user_clinical_data = user_clinical_data,
    measure_name = measure_name,
    volume_columns = volume_columns
  ) %>%
    dplyr::filter(
      LogAge >= x_info$x_min,
      LogAge <= x_info$x_max
    )
  
  y_limits <- get_volume_overlay_y_limits(
    ref_data = ref_data,
    clinical_measure_data = clinical_plot,
    x_info = x_info,
    centile_cols = centile_cols
  )
  
  p <- ggplot2::ggplot(ref_plot, ggplot2::aes(x = LogAge)) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c005]], color = Sex),
      linetype = "dotted",
      linewidth = 1.2,
      alpha = 0.7
    ) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c025]], color = Sex),
      linetype = "dashed",
      linewidth = 1.3,
      alpha = 0.7
    ) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c50]], color = Sex),
      linetype = "solid",
      linewidth = 1.5
    ) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c975]], color = Sex),
      linetype = "dashed",
      linewidth = 1.3,
      alpha = 0.7
    ) +
    
    ggplot2::geom_line(
      ggplot2::aes(y = .data[[centile_cols$c995]], color = Sex),
      linetype = "dotted",
      linewidth = 1.2,
      alpha = 0.7
    ) +
    
    {if (log(280) >= x_info$x_min && log(280) <= x_info$x_max) {
      ggplot2::geom_vline(
        xintercept = log(280),
        linetype = "dotted",
        color = "darkgray",
        linewidth = 1.5
      )
    }} +
    
    ggplot2::geom_point(
      data = clinical_plot,
      ggplot2::aes(x = LogAge, y = Volume),
      inherit.aes = FALSE,
      color = disease_color,
      size = point_size,
      alpha = point_alpha
    ) +
    
    ggplot2::facet_wrap(
      ~ Sex,
      ncol = 2,
      axes = "all_y",
      axis.labels = "all_y"
    ) +
    
    ggplot2::scale_color_manual(values = volume_sex_colors) +
    
    ggplot2::scale_x_continuous(
      breaks = x_info$breaks,
      labels = x_info$labels,
      limits = c(x_info$x_min, x_info$x_max),
      expand = c(0, 0),
      name = NULL
    ) +
    
    ggplot2::scale_y_continuous(
      labels = function(x) x / y_scale$scale,
      expand = ggplot2::expansion(mult = c(0, 0.03)),
      name = NULL
    ) +
    
    ggplot2::annotate(
      "text",
      x = x_info$x_min,
      y = Inf,
      label = y_scale$label,
      hjust = -0.1,
      vjust = 1.2,
      size = 8
    ) +
    
    ggplot2::coord_cartesian(ylim = y_limits) +
    
    ggplot2::theme_classic() +
    ggplot2::theme(
      strip.text = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(
        angle = 45,
        hjust = 1,
        size = 17,
        color = "black"
      ),
      axis.text.y = ggplot2::element_text(
        size = 22,
        color = "black"
      ),
      axis.title = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(
        linewidth = 1.5,
        color = "black"
      ),
      axis.ticks = ggplot2::element_line(
        linewidth = 1.5,
        color = "black"
      ),
      axis.ticks.length = grid::unit(0.28, "cm"),
      legend.position = "none",
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      strip.background = ggplot2::element_blank(),
      plot.title = ggplot2::element_blank(),
      plot.subtitle = ggplot2::element_blank(),
      panel.spacing.x = grid::unit(0.6, "cm"),
      aspect.ratio = 1
    )
  
  g <- ggplot2::ggplotGrob(p)
  
  id <- grep("^axis-l", g$layout$name)
  g$widths[unique(g$layout$l[id])] <- grid::unit(1.8, "cm")
  
  g
}





## ============================================================
## Estimate centile from interpolated centile curve
## ============================================================

estimate_centile_from_curve <- function(y, q_vec, p_vec) {
  
  ok <- is.finite(q_vec) & is.finite(p_vec)
  q_vec <- q_vec[ok]
  p_vec <- p_vec[ok]
  
  if (length(q_vec) < 2 || !is.finite(y)) return(NA_real_)
  
  ord <- order(q_vec)
  q_vec <- q_vec[ord]
  p_vec <- p_vec[ord]
  
  keep <- !duplicated(q_vec)
  q_vec <- q_vec[keep]
  p_vec <- p_vec[keep]
  
  if (length(q_vec) < 2) return(NA_real_)
  
  if (y <= q_vec[1]) return(p_vec[1])
  if (y >= q_vec[length(q_vec)]) return(p_vec[length(p_vec)])
  
  approx(
    x = q_vec,
    y = p_vec,
    xout = y,
    rule = 2
  )$y
}


## ============================================================
## Centile grid helper
## ============================================================

get_volume_centile_grid <- function(ref_data) {
  
  centile_cols <- grep("^Centile_", names(ref_data), value = TRUE)
  
  if (length(centile_cols) == 0) {
    stop("No Centile_ columns found in reference data.")
  }
  
  p_grid <- as.numeric(sub("^Centile_", "", centile_cols)) / 1000
  
  keep_p <- is.finite(p_grid)
  centile_cols <- centile_cols[keep_p]
  p_grid <- p_grid[keep_p]
  
  ord_p <- order(p_grid)
  centile_cols <- centile_cols[ord_p]
  p_grid <- p_grid[ord_p]
  
  dup_p <- duplicated(p_grid)
  
  if (any(dup_p)) {
    centile_cols <- centile_cols[!dup_p]
    p_grid <- p_grid[!dup_p]
  }
  
  list(
    centile_cols = centile_cols,
    p_grid = p_grid
  )
}


## ============================================================
## Compute volume centile scores
## ============================================================

compute_volume_centile_scores <- function(
    user_clinical_data,
    volume_ref_list,
    volume_columns
) {
  
  if (!"LogAge" %in% names(user_clinical_data)) {
    user_clinical_data <- add_derived_volume_variables(user_clinical_data)
  }
  
  if (!"Group" %in% names(user_clinical_data)) {
    stop("user_clinical_data must contain a Group column.")
  }
  
  centile_out_all <- list()
  
  for (measure_name in names(volume_ref_list)) {
    
    ref_data <- volume_ref_list[[measure_name]] %>%
      standardize_volume_sex()
    
    centile_grid <- get_volume_centile_grid(ref_data)
    centile_cols <- centile_grid$centile_cols
    p_grid <- centile_grid$p_grid
    
    clinical_measure_data <- prepare_volume_clinical_measure_data(
      user_clinical_data = user_clinical_data,
      measure_name = measure_name,
      volume_columns = volume_columns
    )
    
    tmp_centile_out_measure <- data.frame()
    
    for (sx in c("Female", "Male")) {
      
      tmp_data_sx <- clinical_measure_data %>%
        dplyr::filter(
          Sex == sx,
          !is.na(Volume),
          Volume != 0,
          is.finite(Volume),
          !is.na(LogAge),
          is.finite(LogAge)
        )
      
      tmp_ref_sx <- ref_data %>%
        dplyr::filter(Sex == sx) %>%
        dplyr::arrange(LogAge)
      
      if (nrow(tmp_data_sx) == 0 || nrow(tmp_ref_sx) == 0) {
        next
      }
      
      pred_q_mat <- sapply(centile_cols, function(cc) {
        approx(
          x = tmp_ref_sx$LogAge,
          y = tmp_ref_sx[[cc]],
          xout = tmp_data_sx$LogAge,
          rule = 2
        )$y
      })
      
      if (is.vector(pred_q_mat)) {
        pred_q_mat <- matrix(pred_q_mat, nrow = nrow(tmp_data_sx))
      }
      
      tmp_centile <- rep(NA_real_, nrow(tmp_data_sx))
      tmp_range_status <- rep(NA_character_, nrow(tmp_data_sx))
      
      for (ii in seq_len(nrow(tmp_data_sx))) {
        
        q_vec <- pred_q_mat[ii, ]
        y_val <- tmp_data_sx$Volume[ii]
        
        tmp_centile[ii] <- estimate_centile_from_curve(
          y = y_val,
          q_vec = q_vec,
          p_vec = p_grid
        )
        
        if (!is.na(y_val) && !all(is.na(q_vec))) {
          if (y_val < min(q_vec, na.rm = TRUE)) {
            tmp_range_status[ii] <- "below_range"
          } else if (y_val > max(q_vec, na.rm = TRUE)) {
            tmp_range_status[ii] <- "above_range"
          } else {
            tmp_range_status[ii] <- "inside_range"
          }
        }
      }
      
      tmp_out <- tmp_data_sx %>%
        dplyr::select(
          dplyr::any_of(c(
            "Subject",
            "Source",
            "Group",
            "PostConceptionDays",
            "LogAge",
            "Sex"
          ))
        )
      
      tmp_out$Measure <- measure_name
      tmp_out$Volume <- tmp_data_sx$Volume
      tmp_out$Centile <- tmp_centile
      tmp_out$Z_score <- qnorm(pmin(pmax(tmp_centile, 1e-6), 1 - 1e-6))
      tmp_out$RangeStatus <- tmp_range_status
      
      tmp_centile_out_measure <- dplyr::bind_rows(
        tmp_centile_out_measure,
        tmp_out
      )
    }
    
    centile_out_all[[measure_name]] <- tmp_centile_out_measure
  }
  
  dplyr::bind_rows(centile_out_all)
}


## ============================================================
## Median centile table helper
## ============================================================

summarize_volume_median_centile <- function(volume_centile_scores) {
  
  tmp_summary <- volume_centile_scores %>%
    dplyr::mutate(
      Centile = as.numeric(Centile),
      Centile = ifelse(Centile > 1, Centile / 100, Centile)
    ) %>%
    dplyr::filter(
      !is.na(Centile),
      is.finite(Centile),
      !is.na(Sex),
      Sex %in% c("Female", "Male"),
      !is.na(Measure)
    ) %>%
    dplyr::group_by(Measure, Sex) %>%
    dplyr::summarise(
      MedianCentile = median(Centile, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::pivot_wider(
      names_from = Sex,
      values_from = MedianCentile,
      names_glue = "{Sex}MedianCentile"
    ) %>%
    dplyr::rename(
      Measurement = Measure
    )
  
  if (!"FemaleMedianCentile" %in% names(tmp_summary)) {
    tmp_summary$FemaleMedianCentile <- NA_real_
  }
  
  if (!"MaleMedianCentile" %in% names(tmp_summary)) {
    tmp_summary$MaleMedianCentile <- NA_real_
  }
  
  tmp_summary %>%
    dplyr::select(
      Measurement,
      FemaleMedianCentile,
      MaleMedianCentile
    )
}




## ============================================================
## Color helpers
## ============================================================

lighten_color <- function(col, amount = 0.45) {
  rgb_col <- grDevices::col2rgb(col) / 255
  rgb_new <- rgb_col + (1 - rgb_col) * amount
  grDevices::rgb(rgb_new[1, ], rgb_new[2, ], rgb_new[3, ])
}

darken_color <- function(col, amount = 0.30) {
  rgb_col <- grDevices::col2rgb(col) / 255
  rgb_new <- rgb_col * (1 - amount)
  grDevices::rgb(rgb_new[1, ], rgb_new[2, ], rgb_new[3, ])
}


## ============================================================
## Half violin helper
## ============================================================

make_half_violin_df <- function(df, x_var, y_var, side_var,
                                width = 0.44, n = 512) {
  
  out <- list()
  idx <- 1
  
  x_levels <- levels(df[[x_var]])
  
  for (dd in x_levels) {
    for (sx in c("Female", "Male")) {
      
      tmp <- df[df[[x_var]] == dd & df[[side_var]] == sx, ]
      y <- tmp[[y_var]]
      y <- y[is.finite(y)]
      
      if (length(y) < 2) next
      
      dens <- stats::density(
        y,
        from = 0,
        n = n,
        na.rm = TRUE
      )
      
      dens_x <- dens$x
      dens_y <- dens$y
      
      if (max(dens_y, na.rm = TRUE) > 0) {
        dens_y <- dens_y / max(dens_y, na.rm = TRUE) * width
      }
      
      x_center <- which(x_levels == dd)
      
      if (sx == "Female") {
        x_poly <- c(rep(x_center, length(dens_x)), rev(x_center - dens_y))
      } else {
        x_poly <- c(rep(x_center, length(dens_x)), rev(x_center + dens_y))
      }
      
      y_poly <- c(dens_x, rev(dens_x))
      
      out[[idx]] <- data.frame(
        Disease = dd,
        Sex = sx,
        x = x_poly,
        y = y_poly,
        DiseaseSex = paste(dd, sx, sep = "_")
      )
      
      idx <- idx + 1
    }
  }
  
  dplyr::bind_rows(out)
}


## ============================================================
## Compute user NDI from centile scores
## ============================================================

compute_user_volume_NDI <- function(
    volume_centile_scores,
    measure_names,
    id_col = "Subject"
) {
  
  disease_name <- unique(volume_centile_scores$Group)
  
  volume_centile_scores %>%
    dplyr::filter(
      Measure %in% measure_names,
      !is.na(Z_score),
      is.finite(Z_score),
      !is.na(Sex),
      Sex %in% c("Female", "Male"),
      !is.na(.data[[id_col]]),
      !is.na(LogAge),
      is.finite(LogAge)
    ) %>%
    dplyr::group_by(
      Sex,
      Subject = .data[[id_col]]
    ) %>%
    dplyr::summarise(
      LogAge = median(LogAge, na.rm = TRUE),
      n_region = dplyr::n(),
      mean_z2 = mean(Z_score^2, na.rm = TRUE),
      rms_z = sqrt(mean_z2),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      Disease = disease_name,
      DiseaseSex = paste(Disease, Sex, sep = "_")
    )
}


## ============================================================
## One-sided disease vs age-matched Healthy permutation test
## ============================================================

perm_test_user_vs_healthy_NDI <- function(
    user_NDI,
    healthy_NDI,
    n_perm = 10000,
    seed = 20260612
) {
  
  set.seed(seed)
  
  disease_name <- unique(user_NDI$Disease)
  
  disease_age_min <- min(user_NDI$LogAge, na.rm = TRUE)
  disease_age_max <- max(user_NDI$LogAge, na.rm = TRUE)
  
  healthy_matched <- healthy_NDI %>%
    dplyr::filter(
      LogAge >= disease_age_min,
      LogAge <= disease_age_max,
      !is.na(rms_z),
      is.finite(rms_z)
    )
  
  y_disease <- user_NDI$rms_z
  y_disease <- y_disease[is.finite(y_disease)]
  
  y_healthy <- healthy_matched$rms_z
  y_healthy <- y_healthy[is.finite(y_healthy)]
  
  if (length(y_disease) < 2 || length(y_healthy) < 2) {
    return(data.frame(
      Disease = disease_name,
      n_disease = length(y_disease),
      n_healthy = length(y_healthy),
      disease_age_min = disease_age_min,
      disease_age_max = disease_age_max,
      median_disease = median(y_disease, na.rm = TRUE),
      median_healthy = median(y_healthy, na.rm = TRUE),
      obs_diff_median = NA_real_,
      p_perm = NA_real_,
      SignifLabel = "",
      SignificantNominal = FALSE
    ))
  }
  
  obs_stat <- median(y_disease, na.rm = TRUE) -
    median(y_healthy, na.rm = TRUE)
  
  y_all <- c(y_disease, y_healthy)
  label_all <- c(
    rep("Disease", length(y_disease)),
    rep("Healthy", length(y_healthy))
  )
  
  perm_stats <- rep(NA_real_, n_perm)
  
  for (bb in seq_len(n_perm)) {
    
    label_perm <- sample(
      label_all,
      size = length(label_all),
      replace = FALSE
    )
    
    perm_stats[bb] <- median(y_all[label_perm == "Disease"], na.rm = TRUE) -
      median(y_all[label_perm == "Healthy"], na.rm = TRUE)
  }
  
  p_perm <- (sum(perm_stats >= obs_stat, na.rm = TRUE) + 1) / (n_perm + 1)
  
  signif_label <- ifelse(
    p_perm < 0.001,
    "***",
    ifelse(
      p_perm < 0.01,
      "**",
      ifelse(
        p_perm < 0.05,
        "*",
        ifelse(p_perm < 0.10, "\u2020", "")
      )
    )
  )
  
  data.frame(
    Disease = disease_name,
    n_disease = length(y_disease),
    n_healthy = length(y_healthy),
    disease_age_min = disease_age_min,
    disease_age_max = disease_age_max,
    median_disease = median(y_disease, na.rm = TRUE),
    median_healthy = median(y_healthy, na.rm = TRUE),
    obs_diff_median = obs_stat,
    p_perm = p_perm,
    SignifLabel = signif_label,
    SignificantNominal = signif_label != ""
  )
}


## ============================================================
## Plot Healthy vs user disease NDI violin
## ============================================================

plot_user_NDI_violin <- function(
    user_NDI,
    healthy_NDI,
    disease_color = NULL
) {
  
  disease_name <- unique(user_NDI$Disease)
  disease_order <- c("Healthy", disease_name)
  
  healthy_plot <- healthy_NDI %>%
    dplyr::mutate(
      Disease = "Healthy",
      DiseaseSex = paste(Disease, Sex, sep = "_")
    )
  
  plot_df <- dplyr::bind_rows(
    healthy_plot,
    user_NDI
  ) %>%
    dplyr::filter(
      !is.na(rms_z),
      is.finite(rms_z),
      !is.na(Sex),
      Sex %in% c("Female", "Male"),
      !is.na(Disease),
      Disease %in% disease_order
    )
  
  plot_df$Disease <- factor(
    plot_df$Disease,
    levels = disease_order
  )
  
  plot_df$Sex <- factor(
    plot_df$Sex,
    levels = c("Female", "Male")
  )
  
  plot_df$DiseaseSex <- paste(
    plot_df$Disease,
    plot_df$Sex,
    sep = "_"
  )
  
  plot_df$x_num <- as.numeric(plot_df$Disease)
  
  disease_base_cols <- setNames(
    scales::hue_pal()(length(disease_order)),
    disease_order
  )
  
  if (!is.null(disease_color)) {
    disease_base_cols[disease_name] <- disease_color
  }
  
  fill_cols <- c()
  
  for (dd in disease_order) {
    
    fill_cols[paste(dd, "Male", sep = "_")] <- darken_color(
      disease_base_cols[dd],
      amount = 0.30
    )
    
    fill_cols[paste(dd, "Female", sep = "_")] <- lighten_color(
      disease_base_cols[dd],
      amount = 0.45
    )
  }
  
  half_violin_df <- make_half_violin_df(
    df = plot_df,
    x_var = "Disease",
    y_var = "rms_z",
    side_var = "Sex",
    width = 0.45
  )
  
  box_half_width <- 0.10
  
  box_df <- plot_df %>%
    dplyr::group_by(Disease, Sex, DiseaseSex, x_num) %>%
    dplyr::summarise(
      q1 = stats::quantile(rms_z, 0.25, na.rm = TRUE),
      q2 = stats::quantile(rms_z, 0.50, na.rm = TRUE),
      q3 = stats::quantile(rms_z, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      xmin = ifelse(Sex == "Female", x_num - box_half_width, x_num),
      xmax = ifelse(Sex == "Female", x_num, x_num + box_half_width)
    )
  
  test_df <- perm_test_user_vs_healthy_NDI(
    user_NDI = user_NDI,
    healthy_NDI = healthy_NDI
  )
  
  shared_y_limits <- c(
    0,
    max(plot_df$rms_z, na.rm = TRUE) * 1.05
  )
  
  star_offset <- 0.08 * diff(shared_y_limits)
  star_top_pad <- 0.08 * diff(shared_y_limits)
  
  common_y_star <- max(half_violin_df$y, na.rm = TRUE) + star_offset
  
  disease_star_df <- test_df %>%
    dplyr::filter(SignificantNominal) %>%
    dplyr::mutate(
      Disease = factor(disease_name, levels = disease_order),
      x_num = as.numeric(Disease),
      y_star = common_y_star,
      label = SignifLabel
    )
  
  shared_y_limits_star <- shared_y_limits
  
  if (nrow(disease_star_df) > 0) {
    shared_y_limits_star[2] <- max(
      shared_y_limits[2],
      common_y_star + star_top_pad
    )
  }
  
  p <- ggplot2::ggplot() +
    ggplot2::geom_polygon(
      data = half_violin_df,
      ggplot2::aes(
        x = x,
        y = y,
        group = interaction(Disease, Sex),
        fill = DiseaseSex
      ),
      color = NA,
      alpha = 0.95
    ) +
    ggplot2::geom_rect(
      data = box_df,
      ggplot2::aes(
        xmin = xmin,
        xmax = xmax,
        ymin = q1,
        ymax = q3,
        fill = DiseaseSex
      ),
      inherit.aes = FALSE,
      color = "black",
      linewidth = 0.5,
      alpha = 0.85
    ) +
    ggplot2::geom_segment(
      data = box_df,
      ggplot2::aes(
        x = xmin,
        xend = xmax,
        y = q2,
        yend = q2
      ),
      inherit.aes = FALSE,
      color = "black",
      linewidth = 0.75
    ) +
    ggplot2::geom_text(
      data = disease_star_df,
      ggplot2::aes(
        x = x_num,
        y = y_star,
        label = label
      ),
      inherit.aes = FALSE,
      size = 9,
      fontface = "bold",
      color = "black"
    ) +
    ggplot2::scale_fill_manual(values = fill_cols) +
    ggplot2::scale_x_continuous(
      breaks = seq_along(disease_order),
      labels = disease_order,
      expand = ggplot2::expansion(mult = c(0.02, 0.02))
    ) +
    ggplot2::coord_cartesian(ylim = shared_y_limits_star) +
    ggplot2::labs(
      x = NULL,
      y = NULL
    ) +
    ggplot2::theme_classic() +
    ggplot2::theme(
      legend.position = "none",
      axis.text.x = ggplot2::element_text(
        angle = 45,
        hjust = 1,
        size = 17,
        color = "black"
      ),
      axis.text.y = ggplot2::element_text(
        size = 22,
        color = "black"
      ),
      axis.title.x = ggplot2::element_blank(),
      axis.title.y = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(
        linewidth = 1.5,
        color = "black"
      ),
      axis.ticks = ggplot2::element_line(
        linewidth = 1.5,
        color = "black"
      ),
      axis.ticks.length = grid::unit(0.28, "cm"),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank()
    )
  
  g <- ggplot2::ggplotGrob(p)
  
  id <- grep("^axis-l", g$layout$name)
  g$widths[unique(g$layout$l[id])] <- grid::unit(1.4, "cm")
  
  list(
    plot = g,
    test = test_df,
    plot_data = plot_df
  )
}
