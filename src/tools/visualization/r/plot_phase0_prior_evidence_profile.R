args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 6) {
  stop("Usage: Rscript plot_phase0_prior_evidence_profile.R <segment_csv> <summary_csv> <output_stem> <title> <subtitle> <threshold_label>")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

segment_csv <- args[[1]]
summary_csv <- args[[2]]
output_stem <- args[[3]]
plot_title <- args[[4]]
plot_subtitle <- args[[5]]
threshold_label <- args[[6]]

segments_df <- read.csv(segment_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_df <- read.csv(summary_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_df <- summary_df[order(summary_df$rank), , drop = FALSE]

display_levels <- rev(summary_df$display_label)
segments_df$display_label <- factor(segments_df$display_label, levels = display_levels)
summary_df$display_label <- factor(summary_df$display_label, levels = display_levels)
segments_df$segment_label <- factor(
  segments_df$segment_label,
  levels = c("Clinical evidence", "Disease specificity", "Consistency bonus", "Mechanistic plausibility")
)
summary_df$status_label <- factor(summary_df$status_label, levels = c("Retained", "Excluded"))

max_total <- max(summary_df$total_score, na.rm = TRUE)
label_padding <- max_total * 0.05

threshold_numeric <- suppressWarnings(as.numeric(sub(".*= ([0-9.]+).*", "\\1", threshold_label)))
if (!is.finite(threshold_numeric)) {
  threshold_numeric <- median(summary_df$total_score, na.rm = TRUE)
}

summary_df$display_index <- match(summary_df$display_label, display_levels)
retained_idx <- summary_df$display_index[summary_df$status_label == "Retained"]
excluded_idx <- summary_df$display_index[summary_df$status_label == "Excluded"]
band_df <- data.frame(
  xmin = c(
    if (length(retained_idx)) min(retained_idx) - 0.5 else NA_real_,
    if (length(excluded_idx)) min(excluded_idx) - 0.5 else NA_real_
  ),
  xmax = c(
    if (length(retained_idx)) max(retained_idx) + 0.5 else NA_real_,
    if (length(excluded_idx)) max(excluded_idx) + 0.5 else NA_real_
  ),
  ymin = -Inf,
  ymax = Inf,
  band = c("Retained band", "Excluded band")
)
band_df <- subset(band_df, is.finite(xmin) & is.finite(xmax))

threshold_x <- if (length(retained_idx)) max(retained_idx) + 0.35 else nrow(summary_df) + 0.35

p <- ggplot(segments_df, aes(x = display_label, y = contribution, fill = segment_label)) +
  geom_rect(
    data = band_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = band),
    inherit.aes = FALSE,
    alpha = 0.28,
    color = NA
  ) +
  geom_col(width = 0.72, color = "#FFFFFF", linewidth = 0.18) +
  geom_hline(
    yintercept = threshold_numeric,
    color = "#B45309",
    linewidth = 0.9,
    linetype = "22"
  ) +
  annotate(
    "text",
    x = threshold_x,
    y = threshold_numeric + 0.03,
    label = threshold_label,
    hjust = 0,
    vjust = 0,
    size = 3.3,
    color = "#92400E",
    fontface = "bold"
  ) +
  geom_point(
    data = summary_df,
    aes(x = display_label, y = total_score, shape = status_label),
    size = 2.2,
    stroke = 0.45,
    fill = "white",
    color = "#334155",
    inherit.aes = FALSE
  ) +
  geom_text(
    data = summary_df,
    aes(x = display_label, y = total_score + label_padding, label = sprintf("%.1f", total_score), color = status_label),
    size = 3.0,
    hjust = 0,
    inherit.aes = FALSE
  ) +
  scale_fill_manual(
    values = c(
      "Retained band" = "#EAF7EE",
      "Excluded band" = "#F4F5F7",
      "Clinical evidence" = "#5B8FF9",
      "Disease specificity" = "#61DDAA",
      "Consistency bonus" = "#9270CA",
      "Mechanistic plausibility" = "#F6BD16"
    ),
    breaks = c("Clinical evidence", "Disease specificity", "Mechanistic plausibility", "Consistency bonus"),
    name = NULL
  ) +
  scale_shape_manual(
    values = c("Retained" = 21, "Excluded" = 4),
    name = NULL
  ) +
  scale_color_manual(
    values = c("Retained" = "#0F172A", "Excluded" = "#64748B"),
    guide = "none"
  ) +
  coord_flip(clip = "off") +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.14))
  ) +
  labs(
    title = plot_title,
    subtitle = plot_subtitle,
    x = NULL,
    y = "Prior evidence score"
  ) +
  guides(
    fill = guide_legend(order = 1, nrow = 1, byrow = TRUE),
    shape = guide_legend(order = 2, nrow = 1, byrow = TRUE, override.aes = list(size = 2.4))
  ) +
  theme_bw(base_family = "sans", base_size = 10.5) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 10.5, hjust = 0.5, color = "#475569"),
    axis.title.x = element_text(size = 11.5, face = "bold"),
    axis.text.y = element_text(size = 8.8, color = "#111827"),
    axis.text.x = element_text(size = 9.5, color = "#374151"),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D1D5DB", linewidth = 0.8),
    legend.position = "top",
    legend.box = "vertical",
    legend.spacing.y = unit(2.5, "mm"),
    legend.key.width = unit(7, "mm"),
    legend.text = element_text(size = 8.8),
    plot.margin = margin(12, 36, 12, 18)
  )

save_plot <- function(plot_obj, output_stem, width, height) {
  pdf(paste0(output_stem, ".pdf"), width = width, height = height, useDingbats = FALSE)
  print(plot_obj)
  dev.off()

  png(paste0(output_stem, ".png"), width = width * 300, height = height * 300, res = 300)
  print(plot_obj)
  dev.off()

  svg(paste0(output_stem, ".svg"), width = width, height = height)
  print(plot_obj)
  dev.off()
}

height_inches <- max(8.4, min(14.6, 4.2 + 0.19 * nrow(summary_df)))
save_plot(p, output_stem, width = 11.8, height = height_inches)
