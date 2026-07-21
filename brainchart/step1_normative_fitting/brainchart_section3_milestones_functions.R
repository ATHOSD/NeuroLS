standardize_milestone_sex <- function(x) {
  x <- as.character(x)
  ifelse(
    x %in% c("F", "Female"),
    "Female",
    ifelse(x %in% c("M", "Male"), "Male", NA_character_)
  )
}

format_milestone_age <- function(post_conception_days) {
  age_years_after_birth <- (post_conception_days - 280) / 365

  if (post_conception_days < 280) {
    paste0(round(post_conception_days / 7, 2), "GA")
  } else if (age_years_after_birth < 1) {
    paste0(round(age_years_after_birth * 12, 2), "mo")
  } else {
    as.character(round(age_years_after_birth, 4))
  }
}

smooth_milestone_median <- function(
    reference_curve,
    df_smooth = 5,
    smooth_start = log(300),
    smooth_apply = log(390),
    blend_width = 0.20) {
  output <- reference_curve

  smoothstep <- function(z) {
    z <- pmin(pmax(z, 0), 1)
    z * z * (3 - 2 * z)
  }

  for (sex in unique(output$Sex)) {
    sex_rows <- output$Sex == sex
    sex_curve <- output[sex_rows, , drop = FALSE]

    if (nrow(sex_curve) < 10) {
      next
    }

    row_order <- order(sex_curve$LogAge)
    sex_curve <- sex_curve[row_order, , drop = FALSE]
    log_age <- sex_curve$LogAge
    median_original <- sex_curve$Centile_50
    blend_weight <- smoothstep(
      (log_age - (smooth_apply - blend_width)) / (2 * blend_width)
    )

    smooth_rows <-
      is.finite(log_age) &
      is.finite(median_original) &
      log_age > smooth_start

    if (sum(smooth_rows) < 4) {
      next
    }

    median_fit <- smooth.spline(
      x = log_age[smooth_rows],
      y = median_original[smooth_rows],
      df = min(df_smooth, sum(smooth_rows) - 1)
    )

    median_smooth <- rep(NA_real_, length(log_age))
    median_smooth[smooth_rows] <- predict(
      median_fit,
      log_age[smooth_rows]
    )$y

    median_final <- median_original
    blend_rows <- smooth_rows & is.finite(median_smooth)
    median_final[blend_rows] <-
      (1 - blend_weight[blend_rows]) * median_original[blend_rows] +
      blend_weight[blend_rows] * median_smooth[blend_rows]

    output_rows <- which(sex_rows)[row_order]
    output[output_rows, "Centile_50"] <- median_final
  }

  output
}

build_milestone_curve <- function(reference_curve, sex, upper_bound) {
  reference_curve$Sex <- standardize_milestone_sex(reference_curve$Sex)
  reference_curve <- reference_curve[
    is.finite(reference_curve$LogAge) &
      reference_curve$LogAge <= upper_bound &
      is.finite(reference_curve$Centile_50) &
      !is.na(reference_curve$Sex),
    ,
    drop = FALSE
  ]

  if (sex == "Pooled") {
    female_curve <- reference_curve[
      reference_curve$Sex == "Female",
      ,
      drop = FALSE
    ]
    male_curve <- reference_curve[
      reference_curve$Sex == "Male",
      ,
      drop = FALSE
    ]
    female_curve <- female_curve[order(female_curve$LogAge), , drop = FALSE]
    male_curve <- male_curve[order(male_curve$LogAge), , drop = FALSE]

    log_age_grid <- sort(unique(c(
      female_curve$LogAge,
      male_curve$LogAge
    )))

    female_median <- approx(
      female_curve$LogAge,
      female_curve$Centile_50,
      xout = log_age_grid,
      rule = 2
    )$y
    male_median <- approx(
      male_curve$LogAge,
      male_curve$Centile_50,
      xout = log_age_grid,
      rule = 2
    )$y

    milestone_curve <- data.frame(
      LogAge = log_age_grid,
      Sex = "Pooled",
      Centile_50 = (female_median + male_median) / 2
    )
  } else {
    milestone_curve <- reference_curve[
      reference_curve$Sex == sex,
      c("LogAge", "Sex", "Centile_50"),
      drop = FALSE
    ]
    milestone_curve <- milestone_curve[
      order(milestone_curve$LogAge),
      ,
      drop = FALSE
    ]
  }

  milestone_curve <- smooth_milestone_median(milestone_curve)
  milestone_curve$Centile_50_standard <-
    milestone_curve$Centile_50 / max(milestone_curve$Centile_50, na.rm = TRUE)
  rownames(milestone_curve) <- NULL
  milestone_curve
}

make_milestone_row <- function(
    reference_set,
    region,
    region_label,
    sex,
    milestone,
    log_age,
    median_value,
    standardized_value,
    velocity = NA_real_,
    standardized_velocity = NA_real_) {
  post_conception_days <- exp(log_age)

  data.frame(
    ReferenceSet = reference_set,
    Region = region,
    RegionLabel = region_label,
    Sex = sex,
    Milestone = milestone,
    LogAge = log_age,
    PostConceptionDays = post_conception_days,
    AgeYearsAfterBirth = (post_conception_days - 280) / 365,
    AgeLabel = format_milestone_age(post_conception_days),
    MedianValue = median_value,
    StandardizedValue = standardized_value,
    VelocityPerLogAge = velocity,
    StandardizedVelocityPerLogAge = standardized_velocity,
    stringsAsFactors = FALSE
  )
}

summarize_curve_milestones <- function(
    milestone_curve,
    reference_set,
    region,
    region_label,
    sex) {
  maximum_index <- which.max(milestone_curve$Centile_50_standard)
  minimum_index <- which.min(milestone_curve$Centile_50_standard)

  maximum_row <- make_milestone_row(
    reference_set,
    region,
    region_label,
    sex,
    "Peak maximum",
    milestone_curve$LogAge[maximum_index],
    milestone_curve$Centile_50[maximum_index],
    milestone_curve$Centile_50_standard[maximum_index]
  )

  minimum_row <- make_milestone_row(
    reference_set,
    region,
    region_label,
    sex,
    "Peak minimum",
    milestone_curve$LogAge[minimum_index],
    milestone_curve$Centile_50[minimum_index],
    milestone_curve$Centile_50_standard[minimum_index]
  )

  log_age_difference <- diff(milestone_curve$LogAge)
  velocity <- diff(milestone_curve$Centile_50) / log_age_difference
  standardized_velocity <-
    diff(milestone_curve$Centile_50_standard) / log_age_difference
  positive_rows <- which(
    is.finite(standardized_velocity) & standardized_velocity > 0
  )

  if (length(positive_rows) == 0) {
    return(rbind(maximum_row, minimum_row))
  }

  velocity_index <- positive_rows[
    which.max(standardized_velocity[positive_rows])
  ]
  adjacent_rows <- c(velocity_index, velocity_index + 1)

  velocity_row <- make_milestone_row(
    reference_set,
    region,
    region_label,
    sex,
    "Max growth velocity",
    mean(milestone_curve$LogAge[adjacent_rows]),
    mean(milestone_curve$Centile_50[adjacent_rows]),
    mean(milestone_curve$Centile_50_standard[adjacent_rows]),
    velocity[velocity_index],
    standardized_velocity[velocity_index]
  )

  rbind(maximum_row, minimum_row, velocity_row)
}

summarize_reference_milestones <- function(
    reference_list,
    reference_set,
    region_labels,
    upper_bound) {
  output <- list()
  output_index <- 1

  for (sex in c("Pooled", "Female", "Male")) {
    for (region in names(reference_list)) {
      milestone_curve <- build_milestone_curve(
        reference_list[[region]],
        sex,
        upper_bound
      )

      output[[output_index]] <- summarize_curve_milestones(
        milestone_curve,
        reference_set,
        region,
        unname(region_labels[[region]]),
        sex
      )
      output_index <- output_index + 1
    }
  }

  do.call(rbind, output)
}
