#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop(
    paste(
      "Usage: Rscript plot_shap_dependence_panels.R",
      "<shap_long.csv> <metadata.json> <output_stem> [theme_json]"
    )
  )
}

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

script_dir <- get_script_dir()
project_lib_candidates <- c(
  Sys.getenv("METABOAGENT_R_LIB", unset = ""),
  file.path(getwd(), ".r_libs"),
  file.path(script_dir, "../../../../.r_libs")
)
project_lib_candidates <- unique(project_lib_candidates[nzchar(project_lib_candidates)])
existing_libs <- project_lib_candidates[file.exists(project_lib_candidates)]
if (length(existing_libs) > 0) {
  .libPaths(c(existing_libs, .libPaths()))
}
source(file.path(get_script_dir(), "font_family_helper.R"))

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(cowplot)
  library(grid)
  library(mgcv)
  library(scales)
})

shap_long_csv <- args[[1]]
metadata_json <- args[[2]]
output_stem <- args[[3]]
theme_json <- ifelse(length(args) >= 4, args[[4]], "")
font_family <- "sans"
text_color <- "#1F2937"
axis_text_color <- "#374151"
grid_color <- "#EDF1F5"
panel_border_color <- "#D8DFE8"
plot_palette <- list(
  main_curve = "#6E3B33",
  confidence_interval = "#D5DFDD",
  positive_fill = "#EEF4F0",
  negative_fill = "#F6E8E5",
  point_low = "#6D9185",
  point_mid = "#F8FAFC",
  point_high = "#6E3B33",
  threshold = "#B46857",
  neutral = "#A39B95"
)

if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- theme$font_family
  }
  if (!is.null(theme$text_color) && nzchar(theme$text_color)) text_color <- theme$text_color
  if (!is.null(theme$axis_text_color) && nzchar(theme$axis_text_color)) axis_text_color <- theme$axis_text_color
  if (!is.null(theme$grid_color) && nzchar(theme$grid_color)) grid_color <- theme$grid_color
  if (!is.null(theme$panel_border_color) && nzchar(theme$panel_border_color)) panel_border_color <- theme$panel_border_color
  if (!is.null(theme$semantic_colors$winner) && nzchar(theme$semantic_colors$winner)) plot_palette$main_curve <- theme$semantic_colors$winner
  if (!is.null(theme$semantic_colors$uncertainty_fill) && nzchar(theme$semantic_colors$uncertainty_fill)) plot_palette$confidence_interval <- theme$semantic_colors$uncertainty_fill
  if (!is.null(theme$semantic_colors$positive) && nzchar(theme$semantic_colors$positive)) {
    plot_palette$positive_fill <- alpha(theme$semantic_colors$positive, 0.12)
    plot_palette$point_low <- theme$semantic_colors$positive
  }
  if (!is.null(theme$semantic_colors$negative) && nzchar(theme$semantic_colors$negative)) {
    plot_palette$negative_fill <- alpha(theme$semantic_colors$negative, 0.12)
    plot_palette$point_high <- theme$semantic_colors$negative
  }
  if (!is.null(theme$semantic_colors$reference) && nzchar(theme$semantic_colors$reference)) plot_palette$neutral <- theme$semantic_colors$reference
  if (!is.null(theme$semantic_colors$highlight) && nzchar(theme$semantic_colors$highlight)) plot_palette$threshold <- theme$semantic_colors$highlight
  if (!is.null(theme$semantic_colors$baseline) && nzchar(theme$semantic_colors$baseline)) {
    plot_palette$main_curve <- theme$semantic_colors$baseline
  }
  if (!is.null(theme$semantic_colors$phase1) && nzchar(theme$semantic_colors$phase1)) {
    plot_palette$point_low <- theme$semantic_colors$phase1
  }
  if (!is.null(theme$semantic_colors$winner) && nzchar(theme$semantic_colors$winner)) {
    plot_palette$point_high <- theme$semantic_colors$winner
  }
}
font_family <- resolve_font_family(font_family)

save_plot <- function(plot_obj, output_stem, width, height) {
  cairo_pdf(paste0(output_stem, ".pdf"), width = width, height = height, family = font_family, bg = "white")
  print(plot_obj)
  dev.off()

  png(paste0(output_stem, ".png"), width = width * 320, height = height * 320, res = 320, bg = "white", type = "cairo")
  print(plot_obj)
  dev.off()

  svg(paste0(output_stem, ".svg"), width = width, height = height, bg = "white")
  print(plot_obj)
  dev.off()
}

format_crossing_label <- function(x) {
  if (!is.finite(x)) {
    return("x = NA")
  }
  paste0("x = ", sprintf("%.3f", x))
}

format_rsq_parse <- function(r_squared) {
  if (!is.finite(r_squared)) {
    return("bolditalic(R)^2 == 'NA'")
  }
  paste0("bolditalic(R)^2 == ", sprintf("%.2f", r_squared))
}

format_p_parse <- function(p_value) {
  if (!is.finite(p_value)) {
    return("bolditalic(p) == 'NA'")
  }
  if (p_value < 0.001) {
    return("bolditalic(p) < 0.001")
  }
  paste0("bolditalic(p) == ", sprintf("%.3f", p_value))
}

deduplicate_crossings <- function(crossings, tolerance) {
  crossings <- crossings[is.finite(crossings)]
  if (length(crossings) <= 1) {
    return(crossings)
  }
  crossings <- sort(crossings)
  kept <- crossings[1]
  for (value in crossings[-1]) {
    if (min(abs(value - kept)) > tolerance) {
      kept <- c(kept, value)
    }
  }
  kept
}

compute_zero_crossings <- function(x, y) {
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]
  y <- y[keep]
  if (length(x) < 3) {
    return(numeric(0))
  }

  order_idx <- order(x)
  x <- x[order_idx]
  y <- y[order_idx]

  exact_idx <- which(abs(y) < 1e-8)
  exact_crossings <- if (length(exact_idx) > 0) x[exact_idx] else numeric(0)

  sign_change_idx <- which(y[-length(y)] * y[-1] < 0)
  interpolated_crossings <- if (length(sign_change_idx) > 0) {
    vapply(
      sign_change_idx,
      function(idx) {
        x0 <- x[idx]
        x1 <- x[idx + 1]
        y0 <- y[idx]
        y1 <- y[idx + 1]
        x0 - y0 * (x1 - x0) / (y1 - y0)
      },
      numeric(1)
    )
  } else {
    numeric(0)
  }

  x_span <- diff(range(x, na.rm = TRUE))
  tolerance <- if (is.finite(x_span) && x_span > 0) x_span * 0.025 else 1e-6
  deduplicate_crossings(c(exact_crossings, interpolated_crossings), tolerance)
}

compute_annotated_zero_crossings <- function(curve_x, curve_y) {
  compute_zero_crossings(curve_x, curve_y)
}

fit_dependence_curve <- function(df) {
  x <- df$feature_value
  y <- df$shap_value
  keep <- is.finite(x) & is.finite(y)
  df <- df[keep, , drop = FALSE]

  if (nrow(df) < 8 || length(unique(df$feature_value)) < 5) {
    return(list(curve = NULL, zero_crossings = numeric(0), edf = NA_real_, smooth_p_value = NA_real_, r_squared = NA_real_))
  }

  unique_n <- length(unique(df$feature_value))
  k_value <- min(7, max(4, floor(unique_n / 4)))
  k_value <- min(k_value, unique_n - 1)
  if (!is.finite(k_value) || k_value < 4) {
    k_value <- min(5, unique_n - 1)
  }

  fit <- tryCatch(
    mgcv::gam(
      shap_value ~ s(feature_value, bs = "cs", k = k_value),
      data = df,
      method = "REML",
      gamma = 1.2
    ),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(list(curve = NULL, zero_crossings = numeric(0), edf = NA_real_, smooth_p_value = NA_real_, r_squared = NA_real_))
  }

  grid_low <- as.numeric(stats::quantile(df$feature_value, probs = 0.01, na.rm = TRUE))
  grid_high <- as.numeric(stats::quantile(df$feature_value, probs = 0.99, na.rm = TRUE))
  if (!is.finite(grid_low) || !is.finite(grid_high) || grid_low >= grid_high) {
    grid_low <- min(df$feature_value, na.rm = TRUE)
    grid_high <- max(df$feature_value, na.rm = TRUE)
  }
  grid_x <- seq(grid_low, grid_high, length.out = 280)
  pred <- tryCatch(
    predict(fit, newdata = data.frame(feature_value = grid_x), se.fit = TRUE, type = "response"),
    error = function(e) NULL
  )

  if (is.null(pred)) {
    return(list(curve = NULL, zero_crossings = numeric(0), edf = NA_real_, smooth_p_value = NA_real_, r_squared = NA_real_))
  }

  fit_summary <- summary(fit)
  smooth_table <- fit_summary$s.table
  edf_value <- if (!is.null(smooth_table) && nrow(smooth_table) >= 1) as.numeric(smooth_table[1, "edf"]) else NA_real_
  p_value <- if (!is.null(smooth_table) && nrow(smooth_table) >= 1) {
    if ("p-value" %in% colnames(smooth_table)) as.numeric(smooth_table[1, "p-value"]) else NA_real_
  } else {
    NA_real_
  }
  r_squared <- if (!is.null(fit_summary$r.sq)) as.numeric(fit_summary$r.sq) else NA_real_

  curve_df <- data.frame(
    feature_value = grid_x,
    fit = as.numeric(pred$fit),
    se = as.numeric(pred$se.fit)
  ) %>%
    mutate(
      lower = fit - 1.96 * se,
      upper = fit + 1.96 * se,
      direction = ifelse(fit >= 0, "Positive", "Negative")
    )

  zero_crossings <- compute_annotated_zero_crossings(curve_df$feature_value, curve_df$fit)
  list(curve = curve_df, zero_crossings = zero_crossings, edf = edf_value, smooth_p_value = p_value, r_squared = r_squared)
}

format_p_value <- function(p_value) {
  if (!is.finite(p_value)) {
    return("NA")
  }
  if (p_value < 0.001) {
    return("< 0.001")
  }
  sprintf("= %.3f", p_value)
}

compute_display_x_limits <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 5) {
    xr <- range(x, na.rm = TRUE)
    pad <- if (diff(xr) > 0) diff(xr) * 0.06 else 0.1
    return(c(xr[1] - pad, xr[2] + pad))
  }

  raw_range <- range(x, na.rm = TRUE)
  q01 <- as.numeric(stats::quantile(x, 0.01, na.rm = TRUE))
  q99 <- as.numeric(stats::quantile(x, 0.99, na.rm = TRUE))
  iqr_value <- stats::IQR(x, na.rm = TRUE)

  # Use robust limits when the feature contains a strong right/left tail.
  use_robust <- (raw_range[2] - q99 > max(iqr_value * 1.5, diff(c(q01, q99)) * 0.2)) ||
    (q01 - raw_range[1] > max(iqr_value * 1.5, diff(c(q01, q99)) * 0.2))

  if (use_robust && q99 > q01) {
    pad <- diff(c(q01, q99)) * 0.06
    return(c(q01 - pad, q99 + pad))
  }

  pad <- if (diff(raw_range) > 0) diff(raw_range) * 0.06 else 0.1
  c(raw_range[1] - pad, raw_range[2] + pad)
}

compute_display_y_limits <- function(y_points, curve_df = NULL) {
  y_points <- y_points[is.finite(y_points)]
  y_curve <- numeric(0)
  if (!is.null(curve_df)) {
    y_curve <- c(curve_df$lower, curve_df$upper)
    y_curve <- y_curve[is.finite(y_curve)]
  }

  y_all <- c(y_points, y_curve)
  if (length(y_all) < 5) {
    yr <- range(c(y_all, 0), na.rm = TRUE)
    pad <- if (diff(yr) > 0) diff(yr) * 0.10 else 0.08
    return(c(yr[1] - pad, yr[2] + pad))
  }

  q01 <- as.numeric(stats::quantile(y_all, 0.01, na.rm = TRUE))
  q99 <- as.numeric(stats::quantile(y_all, 0.99, na.rm = TRUE))
  yr <- range(c(q01, q99, 0), na.rm = TRUE)
  pad <- max(diff(yr) * 0.10, 0.03)
  c(yr[1] - pad, yr[2] + pad)
}

build_single_panel <- function(df, feature_name) {
  fit_obj <- fit_dependence_curve(df)
  curve_df <- fit_obj$curve
  zero_crossings <- fit_obj$zero_crossings
  edf_value <- fit_obj$edf
  smooth_p_value <- fit_obj$smooth_p_value
  r_squared <- fit_obj$r_squared

  x_mid <- stats::median(df$feature_value, na.rm = TRUE)
  y_limits <- compute_display_y_limits(df$shap_value, curve_df)
  label_y <- y_limits[2] - diff(y_limits) * 0.18
  x_limits <- compute_display_x_limits(df$feature_value)
  x_span <- diff(x_limits)
  y_span <- diff(y_limits)
  left_stat_x <- x_limits[1] + x_span * 0.02
  right_stat_x <- x_limits[2] - x_span * 0.02
  top_stat_y <- y_limits[2] - y_span * 0.04
  legend_x <- x_limits[2] - x_span * 0.24
  legend_swatch_xmax <- legend_x + x_span * 0.035
  legend_text_x <- legend_x + x_span * 0.05
  legend_pos_y <- y_limits[2] - y_span * 0.14
  legend_neg_y <- y_limits[2] - y_span * 0.22
  legend_box_xmin <- legend_x - x_span * 0.02
  legend_box_xmax <- legend_x + x_span * 0.19
  legend_box_ymax <- legend_pos_y + y_span * 0.045
  legend_box_ymin <- legend_neg_y - y_span * 0.045

  p <- ggplot(df, aes(x = feature_value, y = shap_value)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = Inf, fill = "#FBF4F6", alpha = 0.90) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0, fill = "#F5F9F5", alpha = 0.95) +
    geom_hline(yintercept = 0, linewidth = 0.62, colour = plot_palette$neutral) +
    geom_point(
      aes(colour = feature_value),
      size = 1.55,
      alpha = 0.78,
      stroke = 0
    ) +
    scale_colour_gradientn(
      colours = c(plot_palette$point_low, plot_palette$point_mid, plot_palette$point_high),
      values = rescale(c(min(df$feature_value, na.rm = TRUE), x_mid, max(df$feature_value, na.rm = TRUE))),
      name = "Feature value"
    ) +
    coord_cartesian(xlim = x_limits, ylim = y_limits, clip = "on") +
    labs(
      title = NULL,
      x = feature_name,
      y = "SHAP value"
    ) +
    theme_bw(base_family = font_family, base_size = 10.8) +
    theme(
      plot.background = element_rect(fill = "white", colour = "white"),
      panel.background = element_rect(fill = "white", colour = "white"),
      panel.grid.major = element_line(colour = grid_color, linewidth = 0.28),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = panel_border_color, linewidth = 0.72),
      axis.title = element_text(size = 10.0, face = "bold", colour = text_color),
      axis.text = element_text(size = 8.7, colour = axis_text_color),
      legend.position = "right",
      legend.title = element_text(size = 8.8, face = "bold", colour = text_color),
      legend.text = element_text(size = 8.2, colour = axis_text_color),
      plot.margin = margin(8, 8, 8, 8)
    )

  if (!is.null(curve_df)) {
    p <- p +
      geom_ribbon(
        data = curve_df,
        aes(x = feature_value, ymin = pmin(fit, 0), ymax = pmax(fit, 0), fill = direction),
        inherit.aes = FALSE,
        alpha = 0.65
      ) +
      geom_ribbon(
        data = curve_df,
        aes(x = feature_value, ymin = lower, ymax = upper),
        inherit.aes = FALSE,
        fill = plot_palette$confidence_interval,
        alpha = 0.26
      ) +
      geom_line(
        data = curve_df,
        aes(x = feature_value, y = fit),
        inherit.aes = FALSE,
        colour = plot_palette$main_curve,
        linewidth = 0.94,
        lineend = "round"
      ) +
      scale_fill_manual(
        values = c(Negative = plot_palette$negative_fill, Positive = plot_palette$positive_fill),
        guide = "none"
      )
  }

  if (length(zero_crossings) > 0) {
    x_offset <- if (is.finite(x_span) && x_span > 0) x_span * 0.028 else 0.08
    crossing_df <- data.frame(
      crossing = sort(zero_crossings),
      stringsAsFactors = FALSE
    )
    crossing_df$rank <- seq_len(nrow(crossing_df))
    crossing_df$label_y <- label_y - (crossing_df$rank - 1) * y_span * 0.085
    lower_bound <- y_limits[1] + y_span * 0.18
    crossing_df$label_y <- pmax(crossing_df$label_y, lower_bound)
    right_tail_cut <- stats::quantile(df$feature_value, 0.84, na.rm = TRUE)
    crossing_df$label_x <- ifelse(
      crossing_df$crossing > right_tail_cut,
      crossing_df$crossing - x_offset,
      crossing_df$crossing + x_offset
    )
    crossing_df$hjust <- ifelse(crossing_df$label_x < crossing_df$crossing, 1, 0)

    p <- p + geom_vline(
      xintercept = crossing_df$crossing,
      linewidth = 0.75,
      linetype = "dashed",
      colour = plot_palette$threshold
    )

    for (i in seq_len(nrow(crossing_df))) {
      p <- p + annotate(
        "label",
        x = crossing_df$label_x[i],
        y = crossing_df$label_y[i],
        label = format_crossing_label(crossing_df$crossing[i]),
        hjust = crossing_df$hjust[i],
        size = 2.95,
        fontface = "bold",
        label.padding = unit(0.18, "lines"),
        fill = "white",
        colour = negative_color,
        label.r = unit(0.08, "lines")
      )
    }
  }

  p <- p +
    annotate(
      "text",
      x = left_stat_x,
      y = top_stat_y,
      label = format_rsq_parse(r_squared),
      hjust = 0,
      vjust = 1,
      parse = TRUE,
      size = 3.0,
      colour = "#4B5563"
    ) +
    annotate(
      "text",
      x = right_stat_x,
      y = top_stat_y,
      label = format_p_parse(smooth_p_value),
      hjust = 1,
      vjust = 1,
      parse = TRUE,
      size = 3.0,
      colour = "#4B5563"
    ) +
    annotate(
      "rect",
      xmin = legend_box_xmin,
      xmax = legend_box_xmax,
      ymin = legend_box_ymin,
      ymax = legend_box_ymax,
      fill = alpha("white", 0.92),
      colour = "#AEB8C3",
      linewidth = 0.35
    ) +
    annotate(
      "rect",
      xmin = legend_x,
      xmax = legend_swatch_xmax,
      ymin = legend_pos_y - y_span * 0.018,
      ymax = legend_pos_y + y_span * 0.018,
      fill = plot_palette$positive_fill,
      colour = "#B8C3B8",
      linewidth = 0.2
    ) +
    annotate(
      "text",
      x = legend_text_x,
      y = legend_pos_y,
      label = "Positive",
      hjust = 0,
      vjust = 0.5,
      size = 2.7,
      colour = "#4B5563"
    ) +
    annotate(
      "rect",
      xmin = legend_x,
      xmax = legend_swatch_xmax,
      ymin = legend_neg_y - y_span * 0.018,
      ymax = legend_neg_y + y_span * 0.018,
      fill = plot_palette$negative_fill,
      colour = "#D0B9BE",
      linewidth = 0.2
    ) +
    annotate(
      "text",
      x = legend_text_x,
      y = legend_neg_y,
      label = "Negative",
      hjust = 0,
      vjust = 0.5,
      size = 2.7,
      colour = "#4B5563"
    )

  summary_row <- data.frame(
    feature = feature_name,
    n = nrow(df),
    median_feature_value = stats::median(df$feature_value, na.rm = TRUE),
    n_zero_crossings = length(zero_crossings),
    zero_crossings = if (length(zero_crossings) > 0) paste(sprintf("%.3f", zero_crossings), collapse = "; ") else NA_character_,
    shap_min = min(df$shap_value, na.rm = TRUE),
    shap_max = max(df$shap_value, na.rm = TRUE),
    r_squared = r_squared,
    gam_edf = edf_value,
    gam_smooth_p_value = smooth_p_value,
    stringsAsFactors = FALSE
  )

  list(plot = p, summary = summary_row)
}

shap_df <- read.csv(shap_long_csv, check.names = FALSE, stringsAsFactors = FALSE)
metadata <- jsonlite::fromJSON(metadata_json)

feature_order <- metadata$feature_order
if (is.null(feature_order) || length(feature_order) == 0) {
  feature_order <- unique(shap_df$feature)
}
feature_order <- feature_order[feature_order %in% shap_df$feature]

shap_df <- shap_df %>% filter(feature %in% feature_order)

panel_results <- lapply(feature_order, function(feature_name) {
  feature_df <- shap_df %>% filter(feature == feature_name)
  build_single_panel(feature_df, feature_name)
})

panels <- lapply(panel_results, function(x) x$plot + theme(legend.position = "none"))
summary_df <- bind_rows(lapply(panel_results, `[[`, "summary"))

title_plot <- ggdraw() +
  draw_label(
    "SHAP Dependence Profiles of the Winner Panel",
    fontface = "bold",
    x = 0.5, y = 0.52, size = 16, colour = "#111827"
  )

blank_panel <- ggdraw()
if (length(panels) == 5) {
  top_row <- plot_grid(plotlist = panels[1:3], ncol = 3, align = "hv")
  bottom_row <- plot_grid(
    blank_panel,
    panels[[4]],
    panels[[5]],
    blank_panel,
    ncol = 4,
    rel_widths = c(0.5, 1, 1, 0.5),
    align = "hv"
  )
  body_grid <- plot_grid(top_row, bottom_row, ncol = 1, rel_heights = c(1, 1))
} else {
  body_grid <- plot_grid(plotlist = panels, ncol = 3, align = "hv")
}
final_plot <- plot_grid(title_plot, body_grid, ncol = 1, rel_heights = c(0.11, 0.89))

save_plot(final_plot, output_stem, width = 15.5, height = 9.6)
write.csv(summary_df, paste0(output_stem, "_summary.csv"), row.names = FALSE)
