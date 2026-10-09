#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop(
    paste(
      "Usage: Rscript plot_shap_binary_combo_preview.R",
      "<shap_long.csv> <feature_importance.csv> <class_ring.csv> <output_stem>"
    )
  )
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(cowplot)
  library(RColorBrewer)
  library(scales)
  library(grid)
})

shap_long_csv <- args[[1]]
feature_importance_csv <- args[[2]]
class_ring_csv <- args[[3]]
output_stem <- args[[4]]

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

simple_beeswarm <- function(x_values, nbins = 40, width = 0.28) {
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

compute_beeswarm_df <- function(df, feature_order) {
  level_order <- rev(feature_order)
  df$feature <- factor(df$feature, levels = level_order)
  df$y_base <- as.numeric(df$feature)

  pieces <- lapply(split(df, df$feature), function(part) {
    part$y <- part$y_base + simple_beeswarm(part$shap_value, nbins = 36, width = 0.24)
    part
  })

  bind_rows(pieces)
}

build_beeswarm_panel <- function(df, panel_title, feature_order, x_limit, show_legend = FALSE) {
  axis_levels <- rev(feature_order)
  df <- compute_beeswarm_df(df, feature_order)

  ggplot(df, aes(x = shap_value, y = y, colour = feature_value_norm)) +
    geom_vline(xintercept = 0, linewidth = 0.70, colour = "#8A8F98") +
    geom_point(size = 1.75, alpha = 0.92, stroke = 0) +
    scale_colour_gradient2(
      low = "#2455A4",
      mid = "#F6F7F8",
      high = "#B2182B",
      midpoint = 0.5,
      limits = c(0, 1),
      name = "Feature value"
    ) +
    scale_y_continuous(
      breaks = seq_along(axis_levels),
      labels = axis_levels,
      expand = expansion(mult = c(0.02, 0.06))
    ) +
    coord_cartesian(xlim = c(-x_limit, x_limit), clip = "off") +
    labs(
      title = panel_title,
      x = "SHAP value for positive class",
      y = NULL
    ) +
    theme_bw(base_family = "sans", base_size = 11) +
    theme(
      plot.background = element_rect(fill = "white", colour = "white"),
      panel.background = element_rect(fill = "white", colour = "white"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(colour = "#EEF1F5", linewidth = 0.28),
      panel.border = element_rect(colour = "#D7DEE8", linewidth = 0.7),
      axis.title.x = element_text(size = 11.8, face = "bold"),
      axis.text.x = element_text(size = 9.8, colour = "#334155"),
      axis.text.y = element_text(size = 10.2, colour = "#111827", face = "bold"),
      plot.title = element_text(size = 13.4, face = "bold", hjust = 0.5, colour = "#111827"),
      legend.position = if (show_legend) "right" else "none",
      legend.title = element_text(size = 9.6, face = "bold"),
      legend.text = element_text(size = 8.8),
      plot.margin = margin(10, 8, 8, 8)
    )
}

build_rose_plot <- function(importance_df, feature_palette) {
  rose_df <- importance_df %>%
    mutate(feature = factor(feature, levels = feature))

  ggplot(rose_df, aes(x = feature, y = proportion, fill = feature)) +
    geom_col(width = 1, colour = "white", linewidth = 0.45) +
    geom_text(
      aes(
        y = proportion + max(proportion) * 0.10,
        label = percent(proportion, accuracy = 0.1)
      ),
      size = 3.1,
      colour = "#111827",
      fontface = "bold"
    ) +
    coord_polar(theta = "x", clip = "off") +
    scale_fill_manual(values = feature_palette, guide = guide_legend(title = "Core features", ncol = 1)) +
    expand_limits(y = max(rose_df$proportion) * 1.28) +
    labs(title = "Global importance rose") +
    theme_void(base_family = "sans", base_size = 11) +
    theme(
      plot.title = element_text(size = 12.8, face = "bold", hjust = 0.5, colour = "#111827"),
      legend.position = "bottom",
      legend.title = element_text(size = 9.6, face = "bold"),
      legend.text = element_text(size = 8.8),
      plot.margin = margin(5, 5, 0, 5)
    )
}

build_ring_plot <- function(class_ring_df, feature_order, feature_palette) {
  class_ids <- sort(unique(class_ring_df$actual_class_id))
  ring_layout <- data.frame(
    actual_class_id = class_ids,
    ring_index = seq_along(class_ids),
    stringsAsFactors = FALSE
  )

  ring_df <- class_ring_df %>%
    left_join(ring_layout, by = "actual_class_id") %>%
    mutate(feature = factor(feature, levels = feature_order)) %>%
    arrange(ring_index, feature)

  ring_df <- ring_df %>%
    group_by(actual_class_id, actual_class_label, ring_index) %>%
    mutate(
      xmax = cumsum(proportion),
      xmin = lag(xmax, default = 0),
      inner = 0.42 + (ring_index - 1) * 0.33,
      outer = inner + 0.24
    ) %>%
    ungroup()

  ring_note <- paste(
    paste0("Inner: ", ring_df$actual_class_label[ring_df$ring_index == min(ring_df$ring_index)][1]),
    paste0("Outer: ", ring_df$actual_class_label[ring_df$ring_index == max(ring_df$ring_index)][1]),
    sep = " | "
  )

  ggplot(ring_df) +
    geom_rect(
      aes(xmin = xmin, xmax = xmax, ymin = inner, ymax = outer, fill = feature),
      colour = "white",
      linewidth = 0.45
    ) +
    coord_polar(theta = "x", clip = "off") +
    scale_fill_manual(values = feature_palette, guide = "none") +
    xlim(0, 1) +
    ylim(0.32, max(ring_df$outer) + 0.10) +
    labs(
      title = "Class-specific contribution rings",
      subtitle = ring_note
    ) +
    theme_void(base_family = "sans", base_size = 11) +
    theme(
      plot.title = element_text(size = 12.8, face = "bold", hjust = 0.5, colour = "#111827"),
      plot.subtitle = element_text(size = 9.5, hjust = 0.5, colour = "#475569"),
      plot.margin = margin(0, 5, 5, 5)
    )
}

shap_long_df <- read.csv(shap_long_csv, check.names = FALSE, stringsAsFactors = FALSE)
importance_df <- read.csv(feature_importance_csv, check.names = FALSE, stringsAsFactors = FALSE)
class_ring_df <- read.csv(class_ring_csv, check.names = FALSE, stringsAsFactors = FALSE)

importance_df <- importance_df %>%
  arrange(rank)

feature_order <- importance_df$feature
class_labels <- unique(shap_long_df[, c("actual_class_id", "actual_class_label")]) %>%
  arrange(actual_class_id)

palette_seed <- c(
  "#4A1028", "#7B241C", "#A93226", "#E67E22", "#F5B041",
  "#F7DC6F", "#76D7C4", "#2874A6", "#1F618D", "#5B5EA6"
)
if (length(feature_order) > length(palette_seed)) {
  palette_seed <- colorRampPalette(brewer.pal(8, "Spectral"))(length(feature_order))
}
feature_palette <- setNames(palette_seed[seq_along(feature_order)], feature_order)

x_limit <- max(abs(shap_long_df$shap_value), na.rm = TRUE) * 1.08

class0_df <- shap_long_df %>% filter(actual_class_id == class_labels$actual_class_id[1])
class1_df <- shap_long_df %>% filter(actual_class_id == class_labels$actual_class_id[2])

panel_a <- build_beeswarm_panel(
  class0_df,
  paste0(class_labels$actual_class_label[1], " samples"),
  feature_order,
  x_limit,
  show_legend = FALSE
)

panel_b <- build_beeswarm_panel(
  class1_df,
  paste0(class_labels$actual_class_label[2], " samples"),
  feature_order,
  x_limit,
  show_legend = TRUE
)

rose_plot <- build_rose_plot(importance_df, feature_palette)
ring_plot <- build_ring_plot(class_ring_df, feature_order, feature_palette)
feature_legend <- get_legend(rose_plot)
rose_plot_nolegend <- rose_plot + theme(legend.position = "none")

summary_col <- plot_grid(
  rose_plot_nolegend,
  ring_plot,
  feature_legend,
  ncol = 1,
  rel_heights = c(1.06, 1.02, 0.54)
)

main_title <- ggdraw() +
  draw_label(
    "SHAP Summary of the Winner Panel",
    fontface = "bold",
    x = 0.5,
    y = 0.5,
    size = 16,
    color = "#111827"
  )

body_plot <- plot_grid(
  panel_a,
  panel_b,
  summary_col,
  labels = c("a", "b", "c"),
  label_size = 14,
  label_fontface = "bold",
  nrow = 1,
  rel_widths = c(1.18, 1.18, 0.98)
)

final_plot <- plot_grid(
  main_title,
  body_plot,
  ncol = 1,
  rel_heights = c(0.12, 0.88)
)

save_plot(final_plot, output_stem, width = 16.2, height = 7.7)
