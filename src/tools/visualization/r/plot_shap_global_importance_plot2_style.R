#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop(
    paste(
      "Usage: Rscript plot_shap_global_importance_plot2_style.R",
      "<shap_long.csv> <feature_importance.csv> <output_stem> [theme_json]"
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

source(file.path(get_script_dir(), "font_family_helper.R"))

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
theme_json <- ifelse(length(args) >= 4, args[[4]], "")
font_family <- "sans"
text_color <- "#1F2937"
axis_text_color <- "#374151"
grid_color <- "#E9EDF2"
panel_border_color <- "#D7DEE8"
bar_fill <- "#D5DFDD"
label_color <- "#64748B"
point_low <- "#6D9185"
point_mid <- "#FFFFFF"
point_high <- "#6E3B33"

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
  if (!is.null(theme$primary_fill) && nzchar(theme$primary_fill)) bar_fill <- theme$primary_fill
  if (!is.null(theme$subtitle_color) && nzchar(theme$subtitle_color)) label_color <- theme$subtitle_color
  if (!is.null(theme$semantic_colors$positive) && nzchar(theme$semantic_colors$positive)) point_low <- theme$semantic_colors$positive
  if (!is.null(theme$semantic_colors$neutral) && nzchar(theme$semantic_colors$neutral)) point_mid <- "#FFFFFF"
  if (!is.null(theme$semantic_colors$negative) && nzchar(theme$semantic_colors$negative)) point_high <- theme$semantic_colors$negative
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

simple_beeswarm <- function(x_values, nbins = 36, width = 0.19) {
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
  arrange(mean_abs_shap)

if (!"feature_display_name" %in% names(shap_df)) {
  shap_df$feature_display_name <- shap_df$feature
}
if (!"feature_display_name" %in% names(importance_df)) {
  importance_df$feature_display_name <- importance_df$feature
}

feature_levels <- importance_df$feature
feature_label_map <- setNames(importance_df$feature_display_name, importance_df$feature)
total_mean_abs <- sum(importance_df$mean_abs_shap, na.rm = TRUE)
n_features <- length(feature_levels)

plot_df <- shap_df %>%
  filter(feature %in% feature_levels) %>%
  mutate(
    feature = factor(feature, levels = feature_levels),
    y_base = as.numeric(feature)
  )

swarm_df <- bind_rows(lapply(split(plot_df, plot_df$feature), function(part) {
  part$y <- part$y_base + simple_beeswarm(part$shap_value, nbins = 40, width = 0.20)
  part
}))

importance_df <- importance_df %>%
  mutate(
    feature = factor(feature, levels = feature_levels),
    y_base = as.numeric(feature),
    proportion_pct = mean_abs_shap / total_mean_abs * 100
  )

shap_range <- range(swarm_df$shap_value, finite = TRUE, na.rm = TRUE)
shap_xlim <- c(min(shap_range[1], -0.01), max(shap_range[2], 0.01))
shap_span <- diff(shap_xlim)
shap_xlim <- c(shap_xlim[1] - shap_span * 0.05, shap_xlim[2] + shap_span * 0.11)

bar_max <- max(importance_df$mean_abs_shap, na.rm = TRUE)
bar_xlim <- c(0, bar_max * 1.28)

transform_to_shap <- function(bar_value) {
  shap_xlim[1] + (bar_value - bar_xlim[1]) / diff(bar_xlim) * diff(shap_xlim)
}

transform_to_bar <- function(shap_value) {
  bar_xlim[1] + (shap_value - shap_xlim[1]) / diff(shap_xlim) * diff(bar_xlim)
}

importance_df <- importance_df %>%
  mutate(
    bar_xmin = transform_to_shap(0),
    bar_xmax = transform_to_shap(mean_abs_shap),
    label_x = transform_to_shap(mean_abs_shap + bar_max * 0.02)
  )

row_bg_df <- data.frame(
  ymin = seq_along(feature_levels) - 0.5,
  ymax = seq_along(feature_levels) + 0.5
)

bar_breaks <- pretty(bar_xlim, n = 4)
bar_breaks <- bar_breaks[bar_breaks >= 0 & bar_breaks <= bar_xlim[2]]

shap_breaks <- pretty(shap_xlim, n = 5)

p <- ggplot() +
  geom_rect(
    data = importance_df,
    aes(xmin = bar_xmin, xmax = bar_xmax, ymin = y_base - 0.30, ymax = y_base + 0.30),
    fill = bar_fill,
    colour = NA,
    alpha = 0.92
  ) +
  geom_point(
    data = swarm_df,
    aes(x = shap_value, y = y, colour = feature_value_norm),
    size = 1.85,
    alpha = 0.82,
    stroke = 0
  ) +
  geom_label(
    data = importance_df,
    aes(x = label_x, y = y_base, label = sprintf("%.1f%%", proportion_pct)),
    hjust = 0,
    vjust = 0.5,
    size = 3.2,
    colour = label_color,
    fontface = "bold",
    fill = alpha("white", 0.68),
    linewidth = 0,
    label.padding = unit(0.10, "lines")
  ) +
  scale_colour_gradient2(
    low = point_low,
    mid = point_mid,
    high = point_high,
    midpoint = 0.5,
    limits = c(0, 1),
    breaks = c(0, 1),
    labels = c("Low", "High"),
    name = "Feature value",
    guide = guide_colourbar(
      direction = "vertical",
      title.position = "right",
      label.position = "right",
      barheight = unit(10.4, "cm"),
      barwidth = unit(0.42, "cm"),
      ticks = FALSE
    )
  ) +
  scale_x_continuous(
    limits = shap_xlim,
    breaks = shap_breaks,
    expand = expansion(mult = c(0, 0)),
    name = "SHAP value (impact on model output)",
    sec.axis = sec_axis(
      transform = ~ transform_to_bar(.),
      breaks = bar_breaks,
      labels = label_number(accuracy = 0.01),
      name = "Mean Absolute SHAP Value"
    )
  ) +
  scale_y_continuous(
    breaks = seq_along(feature_levels),
    labels = unname(feature_label_map[feature_levels]),
    expand = expansion(mult = c(0.03, 0.04))
  ) +
  coord_cartesian(clip = "off") +
  labs(
    title = "Global SHAP Importance of the Winner Panel",
    y = "Features"
  ) +
  theme_minimal(base_family = font_family, base_size = 11) +
  theme(
    plot.background = element_rect(fill = "white", colour = "white"),
    panel.background = element_rect(fill = "white", colour = "white"),
    plot.title = element_text(size = 15.2, face = "bold", hjust = 0.5, colour = text_color, margin = margin(b = 10)),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(colour = grid_color, linewidth = 0.42, linetype = "dashed"),
    axis.title.x.top = element_text(size = 11.5, face = "bold", colour = text_color, margin = margin(b = 7)),
    axis.title.x.bottom = element_text(size = 11.0, face = "bold", colour = text_color, margin = margin(t = 10)),
    axis.title.y = element_text(size = 11.0, face = "bold", colour = text_color, margin = margin(r = 10)),
    axis.text.x.top = element_text(size = 9.1, colour = axis_text_color),
    axis.text.x.bottom = element_text(size = 9.1, colour = axis_text_color),
    axis.text.y = element_text(size = 10.0, colour = text_color),
    axis.line.y.left = element_line(colour = panel_border_color, linewidth = 0.65),
    axis.line.x.top = element_line(colour = panel_border_color, linewidth = 0.65),
    axis.line.x.bottom = element_line(colour = panel_border_color, linewidth = 0.65),
    axis.ticks.y.left = element_line(colour = panel_border_color),
    axis.ticks.x.top = element_line(colour = panel_border_color),
    axis.ticks.x.bottom = element_line(colour = panel_border_color),
    legend.position = "right",
    legend.title = element_text(size = 10.0, face = "bold", colour = text_color, angle = 90, hjust = 0.5, vjust = 0.5),
    legend.text = element_text(size = 8.7, colour = axis_text_color),
    legend.key.height = unit(10.4, "cm"),
    legend.key.width = unit(0.42, "cm"),
    plot.margin = margin(12, 20, 10, 14)
  )

plot_height <- max(5.6, 2.0 + 0.45 * n_features)
save_plot(p, output_stem, width = 8.5, height = plot_height)
