args <- commandArgs(trailingOnly = TRUE)

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

if (length(args) < 4) {
  stop("Usage: Rscript plot_phase3_calibration.R <prediction_csv> <bins_csv> <summary_csv> <output_stem> [theme_json] [title]")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}
source(file.path(get_script_dir(), "font_family_helper.R"))

suppressPackageStartupMessages({
  library(ggplot2)
})

primary_color <- "#8C1D18"
primary_fill <- "#D9A5A0"
neutral_line <- "#B8C0CC"
neutral_segment <- "#D4DAE3"
point_fill <- "#F5F7FA"
text_color <- "#1F2937"
grid_color <- "#E9EDF2"

prediction_csv <- args[[1]]
bins_csv <- args[[2]]
summary_csv <- args[[3]]
output_stem <- args[[4]]
theme_json <- ifelse(length(args) >= 5, args[[5]], "")
title_override <- ifelse(length(args) >= 6, args[[6]], "")
font_family <- "sans"

if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  primary_color <- theme$semantic_colors$winner
  primary_fill <- theme$primary_fill
  neutral_line <- theme$semantic_colors$reference
  neutral_segment <- theme$semantic_colors$neutral
  text_color <- theme$text_color
  grid_color <- theme$grid_color
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- theme$font_family
  }
}
font_family <- resolve_font_family(font_family)

save_plot <- function(plot_obj, output_stem, width, height) {
  cairo_pdf(paste0(output_stem, ".pdf"), width = width, height = height, family = font_family, bg = "white")
  print(plot_obj)
  dev.off()

  png(paste0(output_stem, ".png"), width = width * 300, height = height * 300, res = 300, bg = "white", type = "cairo")
  print(plot_obj)
  dev.off()

  svg(paste0(output_stem, ".svg"), width = width, height = height, bg = "white")
  print(plot_obj)
  dev.off()
}

prediction_df <- read.csv(prediction_csv, check.names = FALSE, stringsAsFactors = FALSE)
bins_df <- read.csv(bins_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_df <- read.csv(summary_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_row <- summary_df[1, , drop = FALSE]

feature_count <- as.integer(summary_row$feature_count[[1]])
assessment_label <- as.character(summary_row$assessment_label[[1]])
brier_score <- as.numeric(summary_row$brier_score[[1]])
ici <- as.numeric(summary_row$ici[[1]])
slope <- as.numeric(summary_row$slope[[1]])
intercept <- as.numeric(summary_row$intercept[[1]])
target_prevalence <- as.numeric(summary_row$target_prevalence[[1]])
n_samples <- if ("n_samples" %in% names(summary_row)) as.numeric(summary_row$n_samples[[1]]) else nrow(prediction_df)
reported_prevalence <- if ("reported_prevalence" %in% names(summary_row)) {
  as.numeric(summary_row$reported_prevalence[[1]])
} else {
  target_prevalence
}
prevalence_label <- if ("prevalence_label" %in% names(summary_row) && nzchar(as.character(summary_row$prevalence_label[[1]]))) {
  as.character(summary_row$prevalence_label[[1]])
} else {
  "Target prevalence"
}

if (!"sample_count" %in% names(bins_df)) {
  bins_df$sample_count <- 1
}

metric_lines <- c(
  paste0("Brier = ", sprintf("%.3f", brier_score)),
  paste0("ICI = ", sprintf("%.3f", ici))
)
if (!is.na(n_samples)) {
  metric_lines <- c(metric_lines, paste0("n = ", as.integer(n_samples)))
}
if (!is.na(slope)) {
  metric_lines <- c(metric_lines, paste0("Slope = ", sprintf("%.3f", slope)))
}
if (!is.na(intercept)) {
  metric_lines <- c(metric_lines, paste0("Intercept = ", sprintf("%.3f", intercept)))
}
if (!is.na(reported_prevalence)) {
  metric_lines <- c(metric_lines, paste0(prevalence_label, " = ", sprintf("%.3f", reported_prevalence)))
}
annotation_label <- paste(metric_lines, collapse = "\n")

prediction_df$observed_label <- as.numeric(prediction_df$observed_label)
prediction_df$predicted_probability <- as.numeric(prediction_df$predicted_probability)

grid_x <- seq(0, 1, length.out = 201)
loess_span <- 0.85

fit_loess_curve <- function(df, grid_values, span_value) {
  fit <- loess(
    observed_label ~ predicted_probability,
    data = df,
    span = span_value,
    degree = 1,
    family = "symmetric",
    surface = "direct",
    control = loess.control(iterations = 1)
  )
  preds <- predict(fit, newdata = data.frame(predicted_probability = grid_values))
  preds <- pmin(pmax(preds, 0), 1)
  preds
}

smooth_values <- tryCatch(
  fit_loess_curve(prediction_df, grid_x, loess_span),
  error = function(e) approx(
    x = bins_df$mean_predicted_probability,
    y = bins_df$observed_event_rate,
    xout = grid_x,
    rule = 2
  )$y
)

set.seed(42)
n_boot <- 400
boot_mat <- matrix(NA_real_, nrow = length(grid_x), ncol = n_boot)
for (boot_id in seq_len(n_boot)) {
  boot_idx <- sample.int(nrow(prediction_df), size = nrow(prediction_df), replace = TRUE)
  boot_df <- prediction_df[boot_idx, , drop = FALSE]
  boot_pred <- tryCatch(
    fit_loess_curve(boot_df, grid_x, loess_span),
    error = function(e) rep(NA_real_, length(grid_x))
  )
  boot_mat[, boot_id] <- boot_pred
}

smooth_lower <- apply(boot_mat, 1, function(x) {
  valid <- x[is.finite(x)]
  if (length(valid) < 20) {
    return(NA_real_)
  }
  quantile(valid, probs = 0.025, na.rm = TRUE)
})
smooth_upper <- apply(boot_mat, 1, function(x) {
  valid <- x[is.finite(x)]
  if (length(valid) < 20) {
    return(NA_real_)
  }
  quantile(valid, probs = 0.975, na.rm = TRUE)
})

smooth_df <- data.frame(
  predicted_probability = grid_x,
  smoothed_observed = smooth_values,
  lower = pmax(pmin(smooth_lower, 1), 0),
  upper = pmax(pmin(smooth_upper, 1), 0)
)

legend_x0 <- 1.02
legend_x1 <- 1.09
legend_text_x <- 1.105
curve_label_y <- 0.90
ideal_label_y <- 0.84
band_label_y <- 0.78

hist_breaks <- seq(0, 1, by = 0.05)
hist_obj <- hist(prediction_df$predicted_probability, breaks = hist_breaks, plot = FALSE)
hist_df <- data.frame(
  x = hist_obj$mids,
  width = diff(hist_obj$breaks)[1],
  count = hist_obj$counts
)
hist_df$height <- if (max(hist_df$count, na.rm = TRUE) > 0) {
  0.085 * hist_df$count / max(hist_df$count, na.rm = TRUE)
} else {
  0
}

ideal_df <- data.frame(
  x = c(0, 1),
  y = c(0, 1)
)

figure_title <- if (nzchar(title_override)) {
  title_override
} else {
  "Calibration Curve of the Winner Panel"
}

p <- ggplot() +
  geom_col(
    data = hist_df,
    aes(x = x, y = height),
    inherit.aes = FALSE,
    width = hist_df$width * 0.92,
    fill = "#E6EBF1",
    color = NA,
    alpha = 0.95
  ) +
  geom_rug(
    data = prediction_df,
    aes(x = predicted_probability),
    inherit.aes = FALSE,
    sides = "b",
    color = "#9AA4B2",
    alpha = 0.32,
    linewidth = 0.22,
    length = grid::unit(0.022, "npc")
  ) +
  geom_line(
    data = ideal_df,
    aes(x = x, y = y),
    linewidth = 0.70,
    color = neutral_line,
    linetype = "22",
    lineend = "round",
    show.legend = FALSE
  ) +
  geom_ribbon(
    data = subset(smooth_df, is.finite(lower) & is.finite(upper)),
    aes(
      x = predicted_probability,
      ymin = lower,
      ymax = upper
    ),
    fill = primary_fill,
    alpha = 0.34,
    show.legend = FALSE
  ) +
  geom_line(
    data = subset(smooth_df, is.finite(smoothed_observed)),
    aes(x = predicted_probability, y = smoothed_observed),
    linewidth = 1.55,
    color = primary_color,
    linetype = "solid",
    lineend = "round",
    na.rm = TRUE,
    show.legend = FALSE
  ) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.2),
    labels = sprintf("%.1f", seq(0, 1, by = 0.2))
  ) +
  scale_y_continuous(
    breaks = seq(0, 1, by = 0.2),
    labels = sprintf("%.1f", seq(0, 1, by = 0.2))
  ) +
  annotate(
    "label",
    x = 0.04,
    y = 0.96,
    label = annotation_label,
    hjust = 0,
    vjust = 1,
    size = 3.45,
    fill = "white",
    color = text_color,
    label.padding = unit(0.20, "lines"),
    label.r = unit(0.12, "lines")
  ) +
  annotate(
    "text",
    x = 0.985,
    y = 0.102,
    label = "Predicted probability distribution",
    hjust = 1,
    vjust = 0,
    size = 3.0,
    color = "#7C8798"
  ) +
  annotate(
    "segment",
    x = legend_x0,
    xend = legend_x1,
    y = curve_label_y,
    yend = curve_label_y,
    linewidth = 1.20,
    color = primary_color,
    lineend = "round"
  ) +
  annotate(
    "text",
    x = legend_text_x,
    y = curve_label_y,
    label = "Calibration curve",
    hjust = 0,
    vjust = 0.5,
    size = 3.15,
    color = primary_color
  ) +
  annotate(
    "segment",
    x = legend_x0,
    xend = legend_x1,
    y = ideal_label_y,
    yend = ideal_label_y,
    linewidth = 0.75,
    color = neutral_line,
    linetype = "22",
    lineend = "round"
  ) +
  annotate(
    "text",
    x = legend_text_x,
    y = ideal_label_y,
    label = "Ideal calibration",
    hjust = 0,
    vjust = 0.5,
    size = 3.05,
    color = "#6B7280"
  ) +
  annotate(
    "rect",
    xmin = legend_x0,
    xmax = legend_x1,
    ymin = band_label_y - 0.010,
    ymax = band_label_y + 0.010,
    fill = primary_fill,
    alpha = 0.34,
    color = NA
  ) +
  annotate(
    "text",
    x = legend_text_x,
    y = band_label_y,
    label = "95% bootstrap band",
    hjust = 0,
    vjust = 0.5,
    size = 3.05,
    color = "#7C3F3B"
  ) +
  labs(
    title = figure_title,
    x = "Mean predicted probability",
    y = "Observed event rate"
  ) +
  theme_bw(base_family = font_family, base_size = 11) +
  theme(
    plot.background = element_rect(fill = "white", color = "white"),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.title = element_text(face = "bold", size = 15.5, hjust = 0.5, color = text_color),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#374151"),
    panel.grid.major = element_line(color = grid_color, linewidth = 0.32),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D7DEE8", linewidth = 0.7),
    legend.position = "none",
    plot.margin = margin(12, 128, 12, 12)
  )

save_plot(p, output_stem, width = 8.2, height = 6.8)
