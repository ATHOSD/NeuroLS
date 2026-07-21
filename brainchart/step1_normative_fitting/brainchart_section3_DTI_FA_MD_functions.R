standardize_dti_sex <- function(x) {
  x <- as.character(x)
  ifelse(
    x %in% c("F", "Female"),
    "Female",
    ifelse(x %in% c("M", "Male"), "Male", NA_character_)
  )
}

prepare_dti_observed_data <- function(data, volume_column) {
  observed <- data.frame(
    Volume = data[[volume_column]],
    LogAge = data$LogAge,
    Sex = standardize_dti_sex(data$Sex),
    Site = data$Site
  )

  observed <- observed[
    is.finite(observed$Volume) &
      observed$Volume > 0 &
      is.finite(observed$LogAge) &
      !is.na(observed$Sex) &
      !is.na(observed$Site) &
      observed$Site != "",
    ,
    drop = FALSE
  ]

  observed$Sex <- factor(observed$Sex, levels = c("Female", "Male"))
  observed$Site <- droplevels(factor(observed$Site))
  rownames(observed) <- NULL
  observed
}

fit_dti_legacy_reference_curve <- function(
    data,
    volume_column,
    reference_site = "baseline",
    mu_df = 20,
    sigma_df = 3) {
  observed <- prepare_dti_observed_data(data, volume_column)
  model_data <- observed
  model_data$Sex <- factor(
    model_data$Sex,
    levels = c("Female", "Male"),
    labels = c("F", "M")
  )

  fit <- gamlss(
    Volume ~ Sex + Site + pb(LogAge, df = mu_df),
    sigma.formula = ~ Sex + pb(LogAge, df = sigma_df),
    family = LOGNO,
    data = model_data,
    control = gamlss.control(n.cyc = 100, trace = FALSE)
  )

  log_age_grid <- seq(log(147), log(280 + 365 * 90), length.out = 300)
  prediction_grid <- data.frame(
    LogAge = rep(log_age_grid, times = 2),
    Sex = factor(rep(c("F", "M"), each = 300), levels = c("F", "M")),
    Site = factor(reference_site, levels = levels(model_data$Site))
  )

  in_range <-
    prediction_grid$LogAge >= min(model_data$LogAge) &
    prediction_grid$LogAge <= max(model_data$LogAge)

  probabilities <- c(0.005, 0.025, 0.25, 0.50, 0.75, 0.975, 0.995)
  centile_columns <- c(
    "Centile_005",
    "Centile_025",
    "Centile_25",
    "Centile_50",
    "Centile_75",
    "Centile_975",
    "Centile_995"
  )

  reference_curve <- data.frame(
    PostConceptionDays = exp(prediction_grid$LogAge),
    LogAge = prediction_grid$LogAge,
    Sex = prediction_grid$Sex
  )

  for (centile_column in centile_columns) {
    reference_curve[[centile_column]] <- NA_real_
  }

  predicted_parameters <- predictAll(
    fit,
    newdata = prediction_grid[in_range, , drop = FALSE],
    type = "response"
  )

  centile_matrix <- sapply(
    probabilities,
    function(probability) {
      qLOGNO(
        probability,
        mu = predicted_parameters$mu,
        sigma = predicted_parameters$sigma
      )
    }
  )

  colnames(centile_matrix) <- centile_columns
  reference_curve[in_range, centile_columns] <- centile_matrix
  reference_curve$Sex <- factor(
    reference_curve$Sex,
    levels = c("F", "M"),
    labels = c("Female", "Male")
  )

  predicted_observed <- predictAll(fit, newdata = model_data, type = "response")
  observed_median <- qLOGNO(
    0.5,
    mu = predicted_observed$mu,
    sigma = predicted_observed$sigma
  )

  reference_data <- model_data
  reference_data$Site <- factor(
    reference_site,
    levels = levels(model_data$Site)
  )
  predicted_reference <- predictAll(
    fit,
    newdata = reference_data,
    type = "response"
  )
  reference_median <- qLOGNO(
    0.5,
    mu = predicted_reference$mu,
    sigma = predicted_reference$sigma
  )

  observed$Volume <- observed$Volume - observed_median + reference_median

  list(
    reference_curve = reference_curve,
    observed_data = observed
  )
}

count_dti_bins <- function(x, breaks) {
  as.integer(
    table(cut(x, breaks = breaks, include.lowest = TRUE, right = TRUE))
  )
}

merge_small_dti_bins <- function(x, breaks, min_n = 10) {
  repeat {
    counts <- count_dti_bins(x, breaks)
    small_bins <- which(counts < min_n)

    if (length(small_bins) == 0 || length(counts) <= 1) {
      break
    }

    small_bin <- small_bins[1]

    if (small_bin == 1) {
      breaks <- breaks[-2]
    } else if (small_bin == length(counts)) {
      breaks <- breaks[-length(breaks)]
    } else if (counts[small_bin - 1] <= counts[small_bin + 1]) {
      breaks <- breaks[-small_bin]
    } else {
      breaks <- breaks[-(small_bin + 1)]
    }
  }

  breaks
}

split_large_dti_bins <- function(
    x,
    breaks,
    min_n = 10,
    target_n = 30,
    max_n = 150) {
  new_breaks <- breaks[1]

  for (bin_index in seq_len(length(breaks) - 1)) {
    left <- breaks[bin_index]
    right <- breaks[bin_index + 1]

    if (bin_index == 1) {
      values <- x[x >= left & x <= right]
    } else {
      values <- x[x > left & x <= right]
    }

    inner_breaks <- numeric(0)

    if (length(values) > max_n) {
      target_bins <- max(2, round(length(values) / target_n))
      target_bins <- min(target_bins, floor(length(values) / min_n))

      if (target_bins >= 2) {
        for (candidate_bins in seq(target_bins, 2, by = -1)) {
          candidate_breaks <- as.numeric(
            quantile(
              values,
              probs = seq(0, 1, length.out = candidate_bins + 1),
              type = 1,
              na.rm = TRUE
            )
          )

          if (length(unique(candidate_breaks)) != length(candidate_breaks)) {
            next
          }

          candidate_counts <- as.integer(
            table(
              cut(
                values,
                breaks = candidate_breaks,
                include.lowest = TRUE,
                right = TRUE
              )
            )
          )

          if (
            length(candidate_counts) == candidate_bins &&
              all(candidate_counts >= min_n)
          ) {
            inner_breaks <- candidate_breaks[
              2:(length(candidate_breaks) - 1)
            ]
            break
          }
        }
      }
    }

    new_breaks <- c(new_breaks, inner_breaks, right)
  }

  sort(unique(new_breaks))
}

merge_narrow_dti_bins <- function(breaks, min_width = 0.01) {
  repeat {
    widths <- diff(breaks)
    narrow_bins <- which(widths < min_width)

    if (length(narrow_bins) == 0 || length(widths) <= 1) {
      break
    }

    narrow_bin <- narrow_bins[1]

    if (narrow_bin == 1) {
      breaks <- breaks[-2]
    } else if (narrow_bin == length(widths)) {
      breaks <- breaks[-length(breaks)]
    } else if (widths[narrow_bin - 1] <= widths[narrow_bin + 1]) {
      breaks <- breaks[-narrow_bin]
    } else {
      breaks <- breaks[-(narrow_bin + 1)]
    }
  }

  breaks
}

make_dti_calibration_breaks <- function(log_age) {
  x <- log_age[is.finite(log_age)]
  breaks <- seq(4.98, 10.41, by = 0.1)

  for (iteration in 1:10) {
    old_breaks <- breaks
    breaks <- merge_small_dti_bins(x, breaks, min_n = 10)
    breaks <- split_large_dti_bins(
      x,
      breaks,
      min_n = 10,
      target_n = 30,
      max_n = 150
    )
    breaks <- merge_small_dti_bins(x, breaks, min_n = 10)

    if (identical(old_breaks, breaks)) {
      break
    }
  }

  merge_narrow_dti_bins(breaks, min_width = 0.01)
}

smooth_dti_legacy_reference <- function(
    reference_curve,
    df_smooth = 12,
    smooth_start = log(300 + 60),
    smooth_apply = log(300 + 90),
    blend_width = 0.05) {
  centile_columns <- c(
    "Centile_005",
    "Centile_025",
    "Centile_25",
    "Centile_50",
    "Centile_75",
    "Centile_975",
    "Centile_995"
  )
  smoothed <- reference_curve

  smoothstep <- function(value) {
    value <- pmin(pmax(value, 0), 1)
    value * value * (3 - 2 * value)
  }

  for (sex in c("Female", "Male")) {
    sex_rows <- which(smoothed$Sex == sex)
    sex_curve <- smoothed[sex_rows, , drop = FALSE]
    order_index <- order(sex_curve$LogAge)
    sex_curve <- sex_curve[order_index, , drop = FALSE]
    log_age <- sex_curve$LogAge
    original_median <- sex_curve$Centile_50
    weight <- smoothstep(
      (log_age - (smooth_apply - blend_width)) / (2 * blend_width)
    )
    median_rows <-
      is.finite(log_age) &
      is.finite(original_median) &
      log_age > smooth_start

    if (sum(median_rows) < 4) {
      next
    }

    median_fit <- smooth.spline(
      x = log_age[median_rows],
      y = original_median[median_rows],
      df = min(df_smooth, sum(median_rows) - 1)
    )
    smoothed_median <- rep(NA_real_, length(log_age))
    smoothed_median[median_rows] <- predict(
      median_fit,
      log_age[median_rows]
    )$y
    final_median <- original_median
    median_use <- median_rows & is.finite(smoothed_median)
    final_median[median_use] <-
      (1 - weight[median_use]) * original_median[median_use] +
      weight[median_use] * smoothed_median[median_use]

    for (centile_column in centile_columns) {
      if (centile_column == "Centile_50") {
        sex_curve[[centile_column]][median_use] <- final_median[median_use]
      } else {
        original_deviation <- sex_curve[[centile_column]] - original_median
        deviation_rows <-
          is.finite(log_age) &
          is.finite(original_deviation) &
          log_age > smooth_start

        if (sum(deviation_rows) >= 4) {
          deviation_fit <- smooth.spline(
            x = log_age[deviation_rows],
            y = original_deviation[deviation_rows],
            df = min(df_smooth, sum(deviation_rows) - 1)
          )
          smoothed_deviation <- rep(NA_real_, length(log_age))
          smoothed_deviation[deviation_rows] <- predict(
            deviation_fit,
            log_age[deviation_rows]
          )$y
          deviation_use <-
            deviation_rows &
            is.finite(smoothed_deviation) &
            is.finite(final_median)
          final_deviation <- original_deviation
          final_deviation[deviation_use] <-
            (1 - weight[deviation_use]) * original_deviation[deviation_use] +
            weight[deviation_use] * smoothed_deviation[deviation_use]
          sex_curve[[centile_column]][deviation_use] <-
            final_median[deviation_use] + final_deviation[deviation_use]
        }
      }
    }

    complete_rows <- complete.cases(sex_curve[centile_columns])
    ordered_centiles <- as.matrix(sex_curve[complete_rows, centile_columns])
    ordered_centiles <- t(apply(ordered_centiles, 1, cummax))
    sex_curve[complete_rows, centile_columns] <- ordered_centiles
    smoothed[sex_rows[order_index], centile_columns] <-
      sex_curve[centile_columns]
  }

  smoothed
}

calibrate_dti_legacy_reference <- function(
    reference_curve,
    observed_data,
    breaks,
    calibration_df = 7) {
  centile_columns <- c(
    "Centile_005",
    "Centile_025",
    "Centile_25",
    "Centile_50",
    "Centile_75",
    "Centile_975",
    "Centile_995"
  )
  calibrated <- reference_curve
  bin_centers <- (breaks[-1] + breaks[-length(breaks)]) / 2
  bin_factor <- cut(
    bin_centers,
    breaks = breaks,
    include.lowest = TRUE,
    right = TRUE
  )
  bin_map <- data.frame(bin = levels(bin_factor), center = bin_centers)

  for (sex in c("Female", "Male")) {
    reference_sex <- calibrated[calibrated$Sex == sex, , drop = FALSE]
    observed_sex <- observed_data[observed_data$Sex == sex, , drop = FALSE]
    reference_sex$bin <- cut(
      reference_sex$LogAge,
      breaks = breaks,
      include.lowest = TRUE,
      right = TRUE
    )
    observed_sex$bin <- cut(
      observed_sex$LogAge,
      breaks = breaks,
      include.lowest = TRUE,
      right = TRUE
    )

    shift_age <- numeric(0)
    shift_value <- numeric(0)
    shift_weight <- numeric(0)
    scale_age <- numeric(0)
    scale_value <- numeric(0)

    for (bin_index in seq_len(nrow(bin_map))) {
      bin <- bin_map$bin[bin_index]
      center <- bin_map$center[bin_index]
      reference_median <- median(
        reference_sex$Centile_50[reference_sex$bin == bin],
        na.rm = TRUE
      )
      observed_median <- median(
        observed_sex$Volume[observed_sex$bin == bin],
        na.rm = TRUE
      )
      reference_iqr <-
        median(
          reference_sex$Centile_75[reference_sex$bin == bin],
          na.rm = TRUE
        ) -
        median(
          reference_sex$Centile_25[reference_sex$bin == bin],
          na.rm = TRUE
        )
      observed_iqr <-
        quantile(
          observed_sex$Volume[observed_sex$bin == bin],
          0.75,
          na.rm = TRUE
        ) -
        quantile(
          observed_sex$Volume[observed_sex$bin == bin],
          0.25,
          na.rm = TRUE
        )

      if (is.finite(reference_median) && is.finite(observed_median)) {
        shift_age <- c(shift_age, center)
        shift_value <- c(shift_value, observed_median - reference_median)
        shift_weight <- c(
          shift_weight,
          ifelse(center <= log(280), 50, 1)
        )
      }

      if (
        is.finite(reference_iqr) &&
          reference_iqr > 0 &&
          is.finite(observed_iqr) &&
          observed_iqr > 0
      ) {
        scale_age <- c(scale_age, center)
        scale_value <- c(scale_value, observed_iqr / reference_iqr)
      }
    }

    if (length(shift_age) >= 4) {
      shift_fit <- smooth.spline(
        x = shift_age,
        y = shift_value,
        w = shift_weight,
        df = min(calibration_df, length(shift_age) - 1)
      )
      predicted_shift <- predict(shift_fit, reference_sex$LogAge)$y
    } else if (length(shift_age) > 0) {
      predicted_shift <- rep(mean(shift_value), nrow(reference_sex))
    } else {
      predicted_shift <- rep(0, nrow(reference_sex))
    }

    if (length(scale_age) >= 4) {
      scale_fit <- smooth.spline(
        x = scale_age,
        y = scale_value,
        df = min(calibration_df, length(scale_age) - 1)
      )
      predicted_scale <- predict(scale_fit, reference_sex$LogAge)$y
    } else if (length(scale_age) > 0) {
      predicted_scale <- rep(mean(scale_value), nrow(reference_sex))
    } else {
      predicted_scale <- rep(1, nrow(reference_sex))
    }

    outside_range <-
      reference_sex$LogAge < min(breaks) |
      reference_sex$LogAge > max(breaks)
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

calculate_dti_coverage_99 <- function(reference_curve, observed_data) {
  results <- lapply(
    c("Female", "Male"),
    function(sex) {
      reference_sex <- reference_curve[
        reference_curve$Sex == sex,
        ,
        drop = FALSE
      ]
      reference_sex <- reference_sex[order(reference_sex$LogAge), ]
      observed_sex <- observed_data[observed_data$Sex == sex, , drop = FALSE]
      lower <- approx(
        reference_sex$LogAge,
        reference_sex$Centile_005,
        observed_sex$LogAge,
        rule = 2
      )$y
      upper <- approx(
        reference_sex$LogAge,
        reference_sex$Centile_995,
        observed_sex$LogAge,
        rule = 2
      )$y
      inside <- observed_sex$Volume >= lower & observed_sex$Volume <= upper

      data.frame(
        Sex = sex,
        N = sum(!is.na(inside)),
        Coverage_99 = mean(inside, na.rm = TRUE)
      )
    }
  )

  do.call(rbind, results)
}

plot_dti_legacy_reference <- function(reference_curve, observed_data) {
  ggplot(reference_curve, aes(x = LogAge)) +
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
    geom_line(aes(y = Centile_50, color = Sex), linewidth = 1.2) +
    geom_point(
      data = observed_data,
      aes(y = Volume, color = Sex),
      size = 0.8,
      alpha = 0.2
    ) +
    facet_wrap(~Sex, ncol = 2) +
    scale_color_manual(values = c(Female = "#BE0E23", Male = "#073E7F")) +
    scale_fill_manual(values = c(Female = "#BE0E23", Male = "#073E7F")) +
    theme_classic() +
    theme(legend.position = "none")
}
