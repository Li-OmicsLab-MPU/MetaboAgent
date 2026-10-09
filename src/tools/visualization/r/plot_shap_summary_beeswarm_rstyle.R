#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop(
    paste(
      "Usage: Rscript plot_shap_summary_beeswarm_rstyle.R",
      "<shap_long.csv> <feature_importance.csv> <output_stem>"
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

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
})

shap_long_csv <- args[[1]]
importance_csv <- args[[2]]
output_stem <- args[[3]]

save_plot <- function(plot_obj, output_stem, width, height) {
  pdf(paste0(output_stem, ".pdf"), width = width, height = height, useDingbats = FALSE, bg = "white")
  print(plot_obj)
  dev.off()

  png(paste0(output_stem, ".png"), width = width * 320, height = height * 320, res = 320, bg = "white")
  print(plot_obj)
  dev.off()

  svg(paste0(output_stem, ".svg"), width = width, height = height, bg = "white")
  print(plot_obj)
  dev.off()
}

simple_beeswarm <- function(x_values, nbins = 42, width = 0.34) {
  x_values <- as.numeric(x_values)
  if (length(x_values) == 0) {
    return(numeric(0))
  }

  x_range <- range(x_values, finite = TRUE)
  if (!all(is.finite(x_range))) {
    return(rep(0, length(x_values)))
  }
  if (isTRUE(all.equal(x_range[1], x_range[2]))) {
    x_range <- c(x_range[1] - 0.1, x_range[2] + 0.1)
  }

  breaks <- seq(x_range[1], x_range[2], length.out = nbins + 1)
  bin_ids <- cut(x_values, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  bin_ids[is.na(bin_ids)] <- ceiling(nbins / 2)
  max_count <- max(tabulate(bin_ids, nbins), 1)

  y_values <- rep(0, length(x_values))
  for (bin_id in sort(unique(bin_ids))) {
    idx <- which(bin_ids == bin_id)
    current_width <- (length(idx) / max_count) * width
    offsets <- seq(-current_width, current_width, length.out = length(idx))
    y_values[idx] <- sample(offsets, length(offsets))
  }
  y_values
}

shap_df <- read.csv(shap_long_csv, check.names = FALSE, stringsAsFactors = FALSE)
importance_df <- read.csv(importance_csv, check.names = FALSE, stringsAsFactors = FALSE) %>%
  arrange(rank)

feature_order <- importance_df$feature
feature_levels <- rev(feature_order)

shap_df <- shap_df %>%
  filter(feature %in% feature_order) %>%
  mutate(
    feature = factor(feature, levels = feature_levels),
    y_base = as.numeric(feature)
  )

split_df <- split(shap_df, shap_df$feature)
swarm_df <- bind_rows(lapply(split_df, function(part) {
  part$y <- part$y_base + simple_beeswarm(part$shap_value, nbins = 44, width = 0.31)
  part
}))

importance_df <- importance_df %>%
  mutate(
    feature = factor(feature, levels = feature_levels),
    y_base = as.numeric(feature)
  )

shap_q_low <- as.numeric(stats::quantile(swarm_df$shap_value, 0.01, na.rm = TRUE))
shap_q_high <- as.numeric(stats::quantile(swarm_df$shap_value, 0.99, na.rm = TRUE))
x_right <- max(shap_q_high, 0.02) * 1.08
x_left <- min(shap_q_low, -0.02) * 1.10
if (!is.finite(x_left) || !is.finite(x_right) || x_left >= x_right) {
  x_left <- min(swarm_df$shap_value, na.rm = TRUE) * 1.05
  x_right <- max(swarm_df$shap_value, na.rm = TRUE) * 1.05
}

bar_gap <- (x_right - x_left) * 0.04
bar_start <- x_left
bar_end <- min(-bar_gap, -0.01)
if (bar_end <= bar_start) {
  bar_end <- x_left + (x_right - x_left) * 0.28
}
max_mean_abs <- max(importance_df$mean_abs_shap, na.rm = TRUE)

importance_df <- importance_df %>%
  mutate(
    bar_xmin = bar_start,
    bar_xmax = bar_start + (mean_abs_shap / max_mean_abs) * (bar_end - bar_start)
  )

row_band_df <- data.frame(
  ymin = seq_len(length(feature_levels)) - 0.5,
  ymax = seq_len(length(feature_levels)) + 0.5
)

sec_transform <- function(x) {
  (x - bar_start) / (bar_end - bar_start) * max_mean_abs
}
sec_inverse <- function(x) {
  bar_start + (x / max_mean_abs) * (bar_end - bar_start)
}

sec_breaks <- pretty(c(0, max_mean_abs), n = 4)
sec_breaks <- sec_breaks[sec_breaks >= 0 & sec_breaks <= max_mean_abs]

p <- ggplot() +
  geom_rect(
    data = row_band_df,
    aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = "#F4F6FB",
    alpha = 0.78
  ) +
  geom_rect(
    data = importance_df,
    aes(xmin = bar_xmin, xmax = bar_xmax, ymin = y_base - 0.36, ymax = y_base + 0.36),
    fill = "#DDE2EE",
    colour = NA,
    alpha = 0.95
  ) +
  geom_vline(xintercept = 0, colour = "#7A8594", linewidth = 0.62) +
  geom_point(
    data = swarm_df,
    aes(x = shap_value, y = y, colour = feature_value_norm),
    size = 1.8,
    alpha = 0.95,
    stroke = 0
  ) +
  scale_colour_gradientn(
    colours = c("#1E88E5", "#7E57C2", "#F5008A"),
    values = c(0, 0.5, 1),
    limits = c(0, 1),
    breaks = c(0, 1),
    labels = c("Low", "High"),
    name = "Feature value"
  ) +
  scale_x_continuous(
    position = "top",
    limits = c(x_left, x_right),
    expand = expansion(mult = c(0, 0)),
    name = "SHAP value (impact on model output)",
    sec.axis = sec_axis(
      transform = ~ sec_transform(.),
      name = "mean(|SHAP value|)",
      breaks = sec_breaks,
      labels = label_number(accuracy = 0.01)
    )
  ) +
  scale_y_continuous(
    breaks = seq_along(feature_levels),
    labels = feature_levels,
    expand = expansion(mult = c(0.02, 0.04))
  ) +
  coord_cartesian(clip = "off") +
  labs(y = "Features") +
  theme_minimal(base_family = "sans", base_size = 10.5) +
  theme(
    plot.background = element_rect(fill = "white", colour = "white"),
    panel.background = element_rect(fill = "white", colour = "white"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(colour = "#E7EBF2", linewidth = 0.32),
    axis.title.x.top = element_text(size = 10.6, face = "bold", margin = margin(b = 6)),
    axis.title.x.bottom = element_text(size = 10.2, face = "bold", margin = margin(t = 8)),
    axis.title.y = element_text(size = 10.2, face = "bold", margin = margin(r = 8)),
    axis.text.x.top = element_text(size = 8.3, colour = "#334155"),
    axis.text.x.bottom = element_text(size = 8.3, colour = "#334155"),
    axis.text.y = element_text(size = 9.0, colour = "#111827"),
    axis.line.x.top = element_line(colour = "#BFC7D5", linewidth = 0.55),
    axis.line.x.bottom = element_line(colour = "#BFC7D5", linewidth = 0.55),
    axis.ticks.x.top = element_line(colour = "#BFC7D5"),
    axis.ticks.x.bottom = element_line(colour = "#BFC7D5"),
    legend.position = "right",
    legend.direction = "vertical",
    legend.title = element_text(size = 9.0, face = "bold"),
    legend.text = element_text(size = 8.0),
    legend.key.height = unit(2.7, "cm"),
    legend.key.width = unit(0.32, "cm"),
    plot.margin = margin(10, 16, 10, 12)
  )

save_plot(p, output_stem, width = 7.8, height = 5.4)
