args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 7) {
  stop("Usage: Rscript plot_phase1_qc_overview.R <retention_csv> <sample_box_csv> <dispersion_csv> <band_csv> <summary_txt> <output_stem> <title>")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
  library(grid)
})

retention_csv <- args[[1]]
sample_box_csv <- args[[2]]
dispersion_csv <- args[[3]]
band_csv <- args[[4]]
summary_txt <- args[[5]]
output_stem <- args[[6]]
plot_title <- args[[7]]

retention_df <- read.csv(retention_csv, check.names = FALSE, stringsAsFactors = FALSE)
sample_df <- read.csv(sample_box_csv, check.names = FALSE, stringsAsFactors = FALSE)
dispersion_df <- read.csv(dispersion_csv, check.names = FALSE, stringsAsFactors = FALSE)
band_df <- read.csv(band_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_lines <- readLines(summary_txt, warn = FALSE, encoding = "UTF-8")
summary_text <- if (length(summary_lines)) paste(summary_lines, collapse = " ") else ""

retention_df$step <- factor(retention_df$step, levels = unique(retention_df$step))
retention_df$series <- factor(retention_df$series, levels = c("Features", "Samples"))
sample_df$Stage <- factor(sample_df$Stage, levels = c("Raw", "Post-PQN"))
dispersion_df$stage <- factor(dispersion_df$stage, levels = c("Raw", "Post-PQN", "Final QC"))

feature_labels <- subset(retention_df, series == "Features")
sample_labels <- subset(retention_df, series == "Samples")
dispersion_highlight <- subset(dispersion_df, !is.na(raw_baseline) & !is.na(post_pqn) & post_pqn < raw_baseline)

make_dispersion_annotation <- function(stage_name, x_target, x_nudge = 0.35, y_nudge = 0) {
  stage_df <- subset(dispersion_df, stage == stage_name)
  if (!nrow(stage_df)) {
    return(NULL)
  }
  idx <- which.min(abs(stage_df$x_log2 - x_target))
  data.frame(
    x = stage_df$x_log2[idx] + x_nudge,
    y = stage_df$normalized_dispersion[idx] + y_nudge,
    label = sprintf("%s%.1f%%", "\u2193", unique(stage_df$reduction_pct)[1]),
    stage = stage_name,
    stringsAsFactors = FALSE
  )
}

annotation_df <- rbind(
  make_dispersion_annotation("Post-PQN", x_target = 12.3, x_nudge = 0.30, y_nudge = 0.010),
  make_dispersion_annotation("Final QC", x_target = 10.5, x_nudge = 0.30, y_nudge = -0.018)
)

waterfall_base <- ggplot(retention_df, aes(x = step, y = retained_count, group = series, color = series)) +
  geom_line(linewidth = 0.95) +
  geom_point(size = 2.4) +
  geom_text(
    data = feature_labels,
    aes(label = label),
    nudge_y = 2.8,
    size = 3.0,
    show.legend = FALSE
  ) +
  geom_text(
    data = sample_labels,
    aes(label = label),
    nudge_y = 4.2,
    size = 3.0,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c("Features" = "#1D4ED8", "Samples" = "#059669")) +
  labs(title = "Retention Waterfall", x = NULL, y = "Retained count", color = NULL) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    axis.text.x = element_text(angle = 35, hjust = 1, color = "#374151"),
    axis.text.y = element_text(color = "#374151"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.35),
    legend.position = "right",
    legend.justification = "center",
    legend.box = "vertical",
    legend.margin = margin(0, 0, 0, 0),
    legend.key.height = unit(5.5, "mm"),
    legend.text = element_text(size = 9.0, color = "#374151"),
    legend.background = element_rect(fill = alpha("white", 0.96), color = NA),
    plot.margin = margin(8, 8, 8, 8)
  )

box_plot <- ggplot(sample_df, aes(x = Stage, y = Total_intensity, fill = Stage)) +
  geom_boxplot(width = 0.58, linewidth = 0.35, outlier.size = 0.8, color = "#6B7280") +
  scale_fill_manual(values = c("Raw" = "#D1D5DB", "Post-PQN" = "#93C5FD")) +
  labs(title = "Sample total intensity", x = NULL, y = NULL) +
  theme_bw(base_family = "sans", base_size = 8) +
  theme(
    plot.title = element_text(size = 8.5, hjust = 0.5, face = "bold"),
    axis.text.x = element_text(size = 6.8, color = "#374151"),
    axis.text.y = element_text(size = 6.6, color = "#374151"),
    panel.grid = element_blank(),
    legend.position = "none",
    plot.background = element_rect(fill = alpha("white", 0.96), color = "#D1D5DB", linewidth = 0.35),
    plot.margin = margin(2, 2, 2, 2)
  )

dispersion_plot <- ggplot() +
  geom_ribbon(
    data = dispersion_highlight,
    aes(x = x_log2, ymin = post_pqn, ymax = raw_baseline, fill = "Normalization gain area"),
    alpha = 0.28
  ) +
  geom_hline(aes(yintercept = 1.0, color = "Raw baseline (=1.0)"), linewidth = 0.8, linetype = "22") +
  geom_line(data = subset(dispersion_df, stage == "Post-PQN"), aes(x = x_log2, y = normalized_dispersion, color = stage), linewidth = 1.0) +
  geom_line(data = subset(dispersion_df, stage == "Final QC"), aes(x = x_log2, y = normalized_dispersion, color = stage), linewidth = 1.0) +
  geom_label(
    data = annotation_df,
    aes(x = x, y = y, label = label, color = stage),
    size = 3.2,
    fontface = "bold",
    linewidth = 0,
    fill = alpha("white", 0.72),
    label.padding = unit(0.12, "lines"),
    show.legend = FALSE
  ) +
  scale_color_manual(
    breaks = c("Raw baseline (=1.0)", "Post-PQN", "Final QC"),
    values = c("Raw baseline (=1.0)" = "#8B97A7", "Post-PQN" = "#2B6CB0", "Final QC" = "#C68A00"),
    labels = c(
      "Raw baseline (=1.0)" = "Raw baseline (=1.0)",
      "Post-PQN" = sprintf("Post-PQN (gain %.1f%%)", unique(subset(dispersion_df, stage == "Post-PQN")$reduction_pct)),
      "Final QC" = sprintf("Final QC (gain %.1f%%)", unique(subset(dispersion_df, stage == "Final QC")$reduction_pct))
    )
  ) +
  scale_fill_manual(values = c("Normalization gain area" = "#93C5FD")) +
  labs(
    title = "Inter-sample KDE Dispersion",
    x = "Log2 intensity",
    y = "Normalized inter-sample dispersion",
    color = NULL,
    fill = NULL
  ) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.title.position = "plot",
    axis.text = element_text(color = "#374151"),
    panel.grid.major = element_line(color = "#E5E7EB", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.justification = "center",
    legend.box = "vertical",
    legend.direction = "vertical",
    legend.margin = margin(0, 0, 0, 0),
    legend.spacing.y = unit(3.2, "mm"),
    legend.text = element_text(size = 9.2, color = "#374151"),
    legend.key.height = unit(6.5, "mm"),
    legend.key.width = unit(8, "mm"),
    plot.margin = margin(8, 8, 8, 8)
  )

dispersion_legend <- get_legend(
  dispersion_plot +
    theme(
      legend.position = "right",
      legend.box.margin = margin(0, 0, 0, 0),
      legend.background = element_rect(fill = alpha("white", 0.96), color = NA)
    ) +
    guides(
      color = guide_legend(order = 1, ncol = 1, byrow = TRUE),
      fill = guide_legend(order = 2, ncol = 1, byrow = TRUE, override.aes = list(alpha = 0.28))
    )
)

dispersion_core <- dispersion_plot +
  theme(
    legend.position = "none",
    plot.margin = margin(8, 8, 8, 8)
  )

dispersion_with_inset <- ggdraw(dispersion_core) +
  draw_plot(box_plot, x = 0.09, y = 0.54, width = 0.29, height = 0.24)

top_row <- plot_grid(
  dispersion_with_inset,
  ggdraw(dispersion_legend),
  ncol = 2,
  align = "h",
  axis = "tb",
  rel_widths = c(1, 0.34)
)

final_plot <- top_row

save_plot <- function(plot_obj, output_stem, width, height) {
  ggsave(paste0(output_stem, ".pdf"), plot_obj, width = width, height = height, device = cairo_pdf, dpi = 300)
  ggsave(paste0(output_stem, ".png"), plot_obj, width = width, height = height, dpi = 300, bg = "white")
  svg(paste0(output_stem, ".svg"), width = width, height = height, bg = "white")
  print(plot_obj)
  dev.off()
}

save_plot(final_plot, output_stem, width = 9.8, height = 5.85)
