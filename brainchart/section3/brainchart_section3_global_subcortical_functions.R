brainchart_centile_columns <- function() {
  c(
    "Centile_005",
    "Centile_025",
    "Centile_25",
    "Centile_50",
    "Centile_75",
    "Centile_975",
    "Centile_995"
  )
}

standardize_brainchart_sex <- function(x) {
  x <- as.character(x)
  ifelse(
    x %in% c("F", "Female"),
    "Female",
    ifelse(x %in% c("M", "Male"), "Male", NA_character_)
  )
}

standardize_reference_curve <- function(
    curve,
    region = NULL,
    region_name = NULL,
    require_full_grid = TRUE) {
  if (!"Sex" %in% names(curve) && "sex" %in% names(curve)) {
    names(curve)[names(curve) == "sex"] <- "Sex"
  }

  legacy_aliases <- c(
    Centile_25 = "Centile_250",
    Centile_50 = "Centile_500",
    Centile_75 = "Centile_750"
  )

  for (legacy_name in names(legacy_aliases)) {
    source_name <- legacy_aliases[[legacy_name]]

    if (!legacy_name %in% names(curve) && source_name %in% names(curve)) {
      curve[[legacy_name]] <- curve[[source_name]]
    }
  }

  required_columns <- c("LogAge", "Sex")
  missing_columns <- setdiff(required_columns, names(curve))

  if (length(missing_columns) > 0) {
    stop(
      "Reference curve is missing columns: ",
      paste(missing_columns, collapse = ", ")
    )
  }

  curve$Sex <- standardize_brainchart_sex(curve$Sex)

  if (anyNA(curve$Sex)) {
    stop("Reference curve contains unrecognized values in Sex.")
  }

  if (require_full_grid) {
    missing_centiles <- setdiff(
      brainchart_centile_columns(),
      names(curve)
    )

    if (length(missing_centiles) > 0) {
      stop(
        "Reference curve is missing centiles: ",
        paste(head(missing_centiles, 5), collapse = ", ")
      )
    }
  }

  if (!is.null(region)) {
    curve$Region <- region
  }

  if (!is.null(region_name)) {
    curve$RegionName <- region_name
  }

  non_centile_columns <- names(curve)[
    !grepl("^Centile_", names(curve))
  ]
  curve <- curve[
    ,
    c(non_centile_columns, brainchart_centile_columns()),
    drop = FALSE
  ]

  curve <- curve[order(match(curve$Sex, c("Female", "Male")), curve$LogAge), ]
  rownames(curve) <- NULL
  curve
}

combine_cerebellum_reference <- function(cortex_curve, white_matter_curve) {
  cortex_curve <- standardize_reference_curve(
    cortex_curve,
    region = "Cerebellum.CortexTransformed",
    region_name = "Cerebellum Cortex"
  )
  white_matter_curve <- standardize_reference_curve(
    white_matter_curve,
    region = "Cerebellum.White.MatterTransformed",
    region_name = "Cerebellum White Matter"
  )

  if (nrow(cortex_curve) != nrow(white_matter_curve)) {
    stop("The two cerebellar reference curves have different row counts.")
  }

  same_age <- isTRUE(all.equal(
    cortex_curve$LogAge,
    white_matter_curve$LogAge,
    tolerance = 1e-10
  ))
  same_sex <- identical(cortex_curve$Sex, white_matter_curve$Sex)

  if (!same_age || !same_sex) {
    stop("The two cerebellar reference curves do not use the same grid.")
  }

  centile_columns <- brainchart_centile_columns()
  metadata_columns <- intersect(
    c("PostConceptionDays", "LogAge", "Sex"),
    names(cortex_curve)
  )
  combined <- cortex_curve[, metadata_columns, drop = FALSE]
  combined$Region <- "CBM"
  combined$RegionName <- "CBM"
  combined[centile_columns] <-
    cortex_curve[centile_columns] +
    white_matter_curve[centile_columns]
  combined
}

calibrate_region_smooth_0428_update <- function(
    region_data,
    observed_data,
    breaks_vec,
    calibration_df = 7) {
  calibrated <- standardize_reference_curve(region_data)
  observed_data$Sex <- standardize_brainchart_sex(observed_data$Sex)
  centile_columns <- grep("^Centile_", names(calibrated), value = TRUE)

  required_centiles <- c("Centile_25", "Centile_50", "Centile_75")
  missing_centiles <- setdiff(required_centiles, centile_columns)

  if (length(missing_centiles) > 0) {
    stop(
      "Calibration requires: ",
      paste(missing_centiles, collapse = ", ")
    )
  }

  min_break <- min(breaks_vec)
  max_break <- max(breaks_vec)
  midpoints <- (breaks_vec[-1] + breaks_vec[-length(breaks_vec)]) / 2
  dummy_cut <- cut(
    midpoints,
    breaks = breaks_vec,
    include.lowest = TRUE,
    right = TRUE
  )
  bin_centers <- data.frame(
    bin = levels(dummy_cut),
    center = midpoints
  )

  for (sex in c("Male", "Female")) {
    reference_sex <- calibrated[calibrated$Sex == sex, ]
    observed_sex <- observed_data[observed_data$Sex == sex, ]

    if (nrow(reference_sex) == 0 || nrow(observed_sex) == 0) {
      next
    }

    reference_sex$bin <- cut(
      reference_sex$LogAge,
      breaks = breaks_vec,
      include.lowest = TRUE,
      right = TRUE
    )
    observed_sex$bin <- cut(
      observed_sex$LogAge,
      breaks = breaks_vec,
      include.lowest = TRUE,
      right = TRUE
    )

    shift_x <- numeric(0)
    shift_y <- numeric(0)
    shift_w <- numeric(0)
    scale_x <- numeric(0)
    scale_y <- numeric(0)
    scale_w <- numeric(0)

    for (bin_index in seq_len(nrow(bin_centers))) {
      bin <- bin_centers$bin[bin_index]
      center <- bin_centers$center[bin_index]
      reference_in_bin <- reference_sex$bin == bin
      observed_in_bin <- observed_sex$bin == bin

      reference_median <- suppressWarnings(median(
        reference_sex$Centile_50[reference_in_bin],
        na.rm = TRUE
      ))
      observed_median <- suppressWarnings(median(
        observed_sex$Volume[observed_in_bin],
        na.rm = TRUE
      ))
      reference_q25 <- suppressWarnings(median(
        reference_sex$Centile_25[reference_in_bin],
        na.rm = TRUE
      ))
      reference_q75 <- suppressWarnings(median(
        reference_sex$Centile_75[reference_in_bin],
        na.rm = TRUE
      ))
      observed_q25 <- suppressWarnings(as.numeric(quantile(
        observed_sex$Volume[observed_in_bin],
        0.25,
        na.rm = TRUE
      )))
      observed_q75 <- suppressWarnings(as.numeric(quantile(
        observed_sex$Volume[observed_in_bin],
        0.75,
        na.rm = TRUE
      )))

      reference_iqr <- reference_q75 - reference_q25
      observed_iqr <- observed_q75 - observed_q25

      if (is.finite(reference_median) && is.finite(observed_median)) {
        shift_x <- c(shift_x, center)
        shift_y <- c(shift_y, observed_median - reference_median)
        shift_w <- c(shift_w, ifelse(center <= log(280), 50, 1))
      }

      if (
        is.finite(reference_iqr) && reference_iqr > 0 &&
          is.finite(observed_iqr) && observed_iqr > 0
      ) {
        scale_x <- c(scale_x, center)
        scale_y <- c(scale_y, observed_iqr / reference_iqr)
        scale_w <- c(scale_w, 1)
      }
    }

    if (length(shift_x) >= 4) {
      shift_fit <- smooth.spline(
        x = shift_x,
        y = shift_y,
        w = shift_w,
        df = min(calibration_df, length(shift_x) - 1)
      )
      predicted_shift <- predict(shift_fit, reference_sex$LogAge)$y
    } else if (length(shift_x) > 0) {
      predicted_shift <- rep(mean(shift_y), nrow(reference_sex))
    } else {
      predicted_shift <- rep(0, nrow(reference_sex))
    }

    if (length(scale_x) >= 4) {
      scale_fit <- smooth.spline(
        x = scale_x,
        y = scale_y,
        w = scale_w,
        df = min(calibration_df, length(scale_x) - 1)
      )
      predicted_scale <- predict(scale_fit, reference_sex$LogAge)$y
    } else if (length(scale_x) > 0) {
      predicted_scale <- rep(mean(scale_y), nrow(reference_sex))
    } else {
      predicted_scale <- rep(1, nrow(reference_sex))
    }

    outside_range <-
      reference_sex$LogAge < min_break |
      reference_sex$LogAge > max_break
    predicted_shift[outside_range] <- 0
    predicted_scale[outside_range] <- 1
    reference_median <- reference_sex$Centile_50

    for (centile_column in centile_columns) {
      reference_sex[[centile_column]] <-
        reference_median +
        predicted_shift +
        predicted_scale *
        (reference_sex[[centile_column]] - reference_median)
    }

    calibrated[
      calibrated$Sex == sex,
      centile_columns
    ] <- reference_sex[centile_columns]
  }

  calibrated
}

prepare_observed_region <- function(data, volume_columns) {
  observed <- data

  if (length(volume_columns) == 1) {
    observed$Volume <- observed[[volume_columns]]
  } else {
    observed$Volume <- rowMeans(
      observed[, volume_columns, drop = FALSE],
      na.rm = FALSE
    )
  }

  observed$Sex <- standardize_brainchart_sex(observed$Sex)
  observed <- observed[
    !is.na(observed$Sex) &
      is.finite(observed$LogAge) &
      is.finite(observed$Volume) &
      observed$Volume != 0,
    ,
    drop = FALSE
  ]
  rownames(observed) <- NULL
  observed
}

fit_gamlss_reference_curve <- function(
    observed_data,
    region,
    region_name,
    selection_method = c("fixed", "bic"),
    npoly_mu = NULL,
    npoly_sigma = NULL,
    p_grid = c(0.005, 0.025, 0.25, 0.50, 0.75, 0.975, 0.995),
    prediction_age_min_days = 147,
    prediction_age_max_days = 280 + 365 * 90,
    prediction_n = 300) {
  selection_method <- match.arg(selection_method)
  model_data <- observed_data[, c("Volume", "LogAge", "Sex")]
  model_data$Sex <- standardize_brainchart_sex(model_data$Sex)
  model_data$Sex <- factor(
    ifelse(model_data$Sex == "Female", "F", "M"),
    levels = c("F", "M")
  )
  model_data <- model_data[
    complete.cases(model_data) &
      is.finite(model_data$Volume) &
      is.finite(model_data$LogAge) &
      model_data$Volume > 0,
    ,
    drop = FALSE
  ]
  model_data$Sex <- droplevels(model_data$Sex)

  if (nlevels(model_data$Sex) != 2) {
    stop(region, " requires observations from both sexes for GAMLSS fitting.")
  }

  fit_one_model <- function(mu_power, sigma_power) {
    gamlss(
      Volume ~ Sex + fp(LogAge, npoly = mu_power),
      sigma.formula = ~ Sex + fp(LogAge, npoly = sigma_power),
      nu.formula = ~ 1,
      family = GG,
      data = model_data,
      control = gamlss.control(n.cyc = 100, trace = FALSE)
    )
  }

  set.seed(31829)
  if (selection_method == "bic") {
    bic_penalty <- log(nrow(model_data))
    best_bic <- Inf
    best_mu <- NA_integer_
    best_sigma <- NA_integer_

    for (candidate_mu in 1:3) {
      for (candidate_sigma in 1:3) {
        candidate_fit <- try(
          fit_one_model(candidate_mu, candidate_sigma),
          silent = TRUE
        )
        if (!inherits(candidate_fit, "try-error")) {
          candidate_bic_result <- try(
            GAIC(candidate_fit, k = bic_penalty),
            silent = TRUE
          )

          if (
            !inherits(candidate_bic_result, "try-error") &&
              is.finite(candidate_bic_result)
          ) {
            candidate_bic <- as.numeric(candidate_bic_result)

            if (candidate_bic < best_bic) {
              best_bic <- candidate_bic
              best_mu <- candidate_mu
              best_sigma <- candidate_sigma
            }
          }
        }
      }
    }

    if (!is.finite(best_bic)) {
      stop("Every GAMLSS candidate failed for ", region, ".")
    }

    npoly_mu <- best_mu
    npoly_sigma <- best_sigma
  } else if (is.null(npoly_mu) || is.null(npoly_sigma)) {
    stop("Fixed GAMLSS selection requires npoly_mu and npoly_sigma.")
  }

  final_fit <- fit_one_model(npoly_mu, npoly_sigma)
  log_age_grid <- seq(
    log(prediction_age_min_days),
    log(prediction_age_max_days),
    length.out = prediction_n
  )
  prediction_grid <- data.frame(
    LogAge = rep(log_age_grid, times = 2),
    Sex = factor(
      rep(c("F", "M"), each = prediction_n),
      levels = c("F", "M")
    )
  )
  in_range <-
    prediction_grid$LogAge >= min(model_data$LogAge) &
    prediction_grid$LogAge <= max(model_data$LogAge)
  centile_columns <- brainchart_centile_columns()
  curve <- data.frame(
    PostConceptionDays = exp(prediction_grid$LogAge),
    LogAge = prediction_grid$LogAge,
    Sex = prediction_grid$Sex,
    Region = region,
    RegionName = region_name
  )

  for (centile_column in centile_columns) {
    curve[[centile_column]] <- NA_real_
  }

  fitted_parameters <- predictAll(
    final_fit,
    newdata = prediction_grid[in_range, , drop = FALSE],
    type = "response"
  )
  centile_matrix <- sapply(
    p_grid,
    function(probability) {
      qGG(
        probability,
        mu = fitted_parameters$mu,
        sigma = fitted_parameters$sigma,
        nu = fitted_parameters$nu
      )
    }
  )
  colnames(centile_matrix) <- centile_columns
  curve[in_range, centile_columns] <- centile_matrix
  curve$Sex <- factor(
    curve$Sex,
    levels = c("F", "M"),
    labels = c("Female", "Male")
  )
  curve <- standardize_reference_curve(curve)

  curve
}

calculate_coverage_99 <- function(reference_curve, observed_data) {
  reference_curve <- standardize_reference_curve(reference_curve)
  observed_data$Sex <- standardize_brainchart_sex(observed_data$Sex)
  results <- vector("list", 2)

  for (sex_index in seq_along(c("Female", "Male"))) {
    sex <- c("Female", "Male")[sex_index]
    reference_sex <- reference_curve[
      reference_curve$Sex == sex &
        is.finite(reference_curve$LogAge) &
        is.finite(reference_curve$Centile_005) &
        is.finite(reference_curve$Centile_995),
      ,
      drop = FALSE
    ]
    observed_sex <- observed_data[
      observed_data$Sex == sex &
        is.finite(observed_data$LogAge) &
        is.finite(observed_data$Volume),
      ,
      drop = FALSE
    ]
    reference_sex <- reference_sex[order(reference_sex$LogAge), ]

    if (nrow(reference_sex) < 2 || nrow(observed_sex) == 0) {
      results[[sex_index]] <- data.frame(
        Sex = sex,
        N = nrow(observed_sex),
        Coverage_99 = NA_real_
      )
      next
    }

    lower <- approx(
      x = reference_sex$LogAge,
      y = reference_sex$Centile_005,
      xout = observed_sex$LogAge,
      rule = 2
    )$y
    upper <- approx(
      x = reference_sex$LogAge,
      y = reference_sex$Centile_995,
      xout = observed_sex$LogAge,
      rule = 2
    )$y
    inside <- observed_sex$Volume >= lower & observed_sex$Volume <= upper

    results[[sex_index]] <- data.frame(
      Sex = sex,
      N = sum(!is.na(inside)),
      Coverage_99 = mean(inside, na.rm = TRUE)
    )
  }

  results <- do.call(rbind, results)
  rbind(
    results,
    data.frame(
      Sex = "Sex-balanced",
      N = sum(results$N),
      Coverage_99 = mean(results$Coverage_99, na.rm = TRUE)
    )
  )
}

get_age_labels <- function() {
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

  list(breaks = log(ages$days), labels = ages$label)
}

plot_region_truncated_DTI_0502 <- function(
    reference_curve,
    observed_data = NULL,
    upper_bound = NULL,
    sex_colors = c(Female = "#BE0E23", Male = "#073E7F")) {
  reference_curve <- standardize_reference_curve(reference_curve)

  if (!is.null(upper_bound)) {
    reference_curve <- reference_curve[
      reference_curve$LogAge <= upper_bound,
      ,
      drop = FALSE
    ]
  }

  if (!is.null(observed_data)) {
    observed_data$Sex <- standardize_brainchart_sex(observed_data$Sex)

    if (!is.null(upper_bound)) {
      observed_data <- observed_data[
        observed_data$LogAge <= upper_bound,
        ,
        drop = FALSE
      ]
    }
  }

  age_axis <- get_age_labels()
  x_min <- min(reference_curve$LogAge, na.rm = TRUE)
  x_max <- if (is.null(upper_bound)) {
    max(reference_curve$LogAge, na.rm = TRUE)
  } else {
    upper_bound
  }

  plot <- ggplot(reference_curve, aes(x = LogAge)) +
    geom_ribbon(
      aes(ymin = Centile_005, ymax = Centile_995, fill = Sex),
      alpha = 0.10
    ) +
    geom_ribbon(
      aes(ymin = Centile_025, ymax = Centile_975, fill = Sex),
      alpha = 0.15
    ) +
    geom_ribbon(
      aes(ymin = Centile_25, ymax = Centile_75, fill = Sex),
      alpha = 0.25
    ) +
    geom_line(
      aes(y = Centile_005, color = Sex),
      linetype = "dotted",
      linewidth = 1.2,
      alpha = 0.7
    ) +
    geom_line(
      aes(y = Centile_025, color = Sex),
      linetype = "dashed",
      linewidth = 1.3,
      alpha = 0.7
    ) +
    geom_line(
      aes(y = Centile_25, color = Sex),
      linetype = "twodash",
      linewidth = 1.4,
      alpha = 0.8
    ) +
    geom_line(
      aes(y = Centile_50, color = Sex),
      linetype = "solid",
      linewidth = 1.5
    ) +
    geom_line(
      aes(y = Centile_75, color = Sex),
      linetype = "twodash",
      linewidth = 1.4,
      alpha = 0.8
    ) +
    geom_line(
      aes(y = Centile_975, color = Sex),
      linetype = "dashed",
      linewidth = 1.3,
      alpha = 0.7
    ) +
    geom_line(
      aes(y = Centile_995, color = Sex),
      linetype = "dotted",
      linewidth = 1.2,
      alpha = 0.7
    ) +
    geom_vline(
      xintercept = log(280),
      linetype = "dotted",
      color = "darkgray",
      linewidth = 1.5
    ) +
    facet_wrap(~Sex, ncol = 2, axes = "all_y", axis.labels = "all_y") +
    scale_color_manual(values = sex_colors) +
    scale_fill_manual(values = sex_colors) +
    scale_x_continuous(
      breaks = age_axis$breaks,
      labels = age_axis$labels,
      limits = c(x_min, x_max),
      expand = c(0, 0),
      name = NULL
    ) +
    scale_y_continuous(
      labels = function(x) x / 1e4,
      expand = expansion(mult = c(0, 0.03)),
      name = NULL
    ) +
    annotate(
      "text",
      x = x_min,
      y = Inf,
      label = "×10^4",
      hjust = -0.1,
      vjust = 1.2,
      size = 8
    ) +
    theme_classic() +
    theme(
      strip.text = element_text(size = 28, face = "bold"),
      axis.text.x = element_text(
        angle = 45,
        hjust = 1,
        size = 17,
        color = "black"
      ),
      axis.text.y = element_text(size = 22, color = "black"),
      axis.title = element_blank(),
      axis.line = element_line(linewidth = 1.5, color = "black"),
      axis.ticks = element_line(linewidth = 1.5, color = "black"),
      axis.ticks.length = grid::unit(0.28, "cm"),
      legend.position = "none",
      panel.grid = element_blank(),
      panel.border = element_blank(),
      strip.background = element_blank(),
      panel.spacing.x = grid::unit(0.6, "cm")
    )

  if (!is.null(observed_data) && nrow(observed_data) > 0) {
    plot <- plot +
      geom_point(
        data = observed_data,
        aes(
          x = LogAge,
          y = Volume,
          color = Sex,
          shape = ind_T2,
          alpha = ind_T2
        ),
        size = 1
      ) +
      scale_shape_manual(values = c(`FALSE` = 19, `TRUE` = 17)) +
      scale_alpha_manual(values = c(`FALSE` = 0.2, `TRUE` = 0.9))
  }

  plot
}
