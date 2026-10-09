args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Usage: Rscript plot_phase0_prior_profile_alternatives.R <summary_csv> <segments_csv> <output_dir>")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(pheatmap)
})

summary_csv <- args[[1]]
segments_csv <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

summary_df <- read.csv(summary_csv, check.names = FALSE, stringsAsFactors = FALSE)
segments_df <- read.csv(segments_csv, check.names = FALSE, stringsAsFactors = FALSE)

wrap_label <- function(x, width = 26) {
  if (!nzchar(x)) return(x)
  paste(strwrap(x, width = width), collapse = "\n")
}

save_gg_bundle <- function(plot_obj, output_stem, width, height) {
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

summary_df <- summary_df[order(summary_df$rank), , drop = FALSE]
summary_df$display_label <- vapply(summary_df$display_label, wrap_label, character(1))
segments_df$display_label <- vapply(segments_df$display_label, wrap_label, character(1))

dimension_order <- c(
  "Mechanistic plausibility",
  "Disease specificity",
  "Clinical evidence",
  "Consistency bonus"
)
dimension_short <- c(
  "Mechanistic plausibility" = "Mechanistic\nplausibility",
  "Disease specificity" = "Disease\nspecificity",
  "Clinical evidence" = "Clinical\nevidence",
  "Consistency bonus" = "Consistency\nbonus"
)

segments_df$segment_label <- factor(segments_df$segment_label, levels = dimension_order)
summary_df$status_label <- factor(summary_df$status_label, levels = c("Retained", "Excluded"))

matrix_wide <- reshape(
  segments_df[, c("display_label", "segment_label", "contribution")],
  idvar = "display_label",
  timevar = "segment_label",
  direction = "wide"
)
colnames(matrix_wide) <- sub("^contribution\\.", "", colnames(matrix_wide))
for (dim_name in dimension_order) {
  if (!(dim_name %in% colnames(matrix_wide))) {
    matrix_wide[[dim_name]] <- 0
  }
}
matrix_wide <- merge(
  summary_df[, c("display_label", "total_score", "status_label", "rank")],
  matrix_wide,
  by = "display_label",
  all.x = TRUE,
  sort = FALSE
)
matrix_wide <- matrix_wide[order(matrix_wide$rank), , drop = FALSE]

score_matrix <- as.matrix(matrix_wide[, dimension_order, drop = FALSE])
rownames(score_matrix) <- matrix_wide$display_label
score_matrix[is.na(score_matrix)] <- 0

row_anno <- data.frame(
  TotalScore = matrix_wide$total_score,
  Status = matrix_wide$status_label
)
rownames(row_anno) <- matrix_wide$display_label

heatmap_path <- file.path(output_dir, "phase0_prior_evidence_heatmap_clustered.pdf")
heatmap_colors <- colorRampPalette(c("#FFFFFF", "#DCEBFA", "#7AA6D6", "#1F4E79"))(120)
anno_colors <- list(
  Status = c("Retained" = "#A7D7B5", "Excluded" = "#D8DCE3"),
  TotalScore = colorRampPalette(c("#F8FBFF", "#C6DBEF", "#6BAED6", "#08519C"))(100)
)

pdf(heatmap_path, width = 8.8, height = max(8.0, min(13.5, 3.4 + 0.18 * nrow(score_matrix))))
pheatmap(
  mat = score_matrix,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  color = heatmap_colors,
  border_color = "#F3F4F6",
  cellwidth = 28,
  cellheight = 11,
  fontsize = 8.5,
  fontsize_row = 8.0,
  fontsize_col = 9.5,
  angle_col = 0,
  main = "Prior Evidence Heatmap With Clustering",
  annotation_row = row_anno,
  annotation_colors = anno_colors,
  treeheight_row = 42,
  show_rownames = TRUE,
  show_colnames = TRUE,
  legend = TRUE
)
dev.off()
png(sub("\\.pdf$", ".png", heatmap_path), width = 8.8 * 300, height = max(8.0, min(13.5, 3.4 + 0.18 * nrow(score_matrix))) * 300, res = 300)
pheatmap(
  mat = score_matrix,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  color = heatmap_colors,
  border_color = "#F3F4F6",
  cellwidth = 28,
  cellheight = 11,
  fontsize = 8.5,
  fontsize_row = 8.0,
  fontsize_col = 9.5,
  angle_col = 0,
  main = "Prior Evidence Heatmap With Clustering",
  annotation_row = row_anno,
  annotation_colors = anno_colors,
  treeheight_row = 42,
  show_rownames = TRUE,
  show_colnames = TRUE,
  legend = TRUE
)
dev.off()
svg(sub("\\.pdf$", ".svg", heatmap_path), width = 8.8, height = max(8.0, min(13.5, 3.4 + 0.18 * nrow(score_matrix))))
pheatmap(
  mat = score_matrix,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  color = heatmap_colors,
  border_color = "#F3F4F6",
  cellwidth = 28,
  cellheight = 11,
  fontsize = 8.5,
  fontsize_row = 8.0,
  fontsize_col = 9.5,
  angle_col = 0,
  main = "Prior Evidence Heatmap With Clustering",
  annotation_row = row_anno,
  annotation_colors = anno_colors,
  treeheight_row = 42,
  show_rownames = TRUE,
  show_colnames = TRUE,
  legend = TRUE
)
dev.off()

bubble_df <- merge(
  segments_df,
  summary_df[, c("display_label", "rank", "total_score", "status_label")],
  by = "display_label",
  suffixes = c("", "_summary")
)
bubble_df <- subset(bubble_df, contribution > 0)
bubble_df$display_label <- factor(
  bubble_df$display_label,
  levels = rev(summary_df$display_label)
)
bubble_df$segment_display <- factor(
  dimension_short[as.character(bubble_df$segment_label)],
  levels = dimension_short[dimension_order]
)

retained_n <- sum(summary_df$status_label == "Retained")
total_n <- nrow(summary_df)
band_df <- data.frame(
  xmin = c(0.5, 0.5),
  xmax = c(length(dimension_order) + 0.5, length(dimension_order) + 0.5),
  ymin = c(total_n - retained_n + 0.5, 0.5),
  ymax = c(total_n + 0.5, total_n - retained_n + 0.5),
  band = c("Retained", "Excluded")
)

bubble_plot <- ggplot() +
  geom_rect(
    data = band_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = band),
    inherit.aes = FALSE,
    alpha = 0.32,
    color = NA
  ) +
  geom_point(
    data = bubble_df,
    aes(x = segment_display, y = display_label, size = contribution, color = total_score),
    alpha = 0.96
  ) +
  scale_fill_manual(values = c("Retained" = "#ECF8F0", "Excluded" = "#F3F4F6"), guide = "none") +
  scale_size_continuous(
    range = c(1.8, 10.5),
    breaks = c(0.5, 1, 2, 4, 6),
    name = "Contribution"
  ) +
  scale_color_gradientn(
    colours = c("#D7E3EF", "#A8C8E3", "#6FA8DC", "#2E75B6", "#0B3C78"),
    name = "Total score"
  ) +
  labs(
    title = "Prior Evidence Bubble Matrix",
    subtitle = "Bubble size encodes per-dimension contribution; color encodes total prior score",
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 10.5, hjust = 0.5, color = "#475569"),
    axis.text.x = element_text(size = 9.5, color = "#374151"),
    axis.text.y = element_text(size = 8.4, color = "#111827"),
    panel.grid.major = element_line(color = "#E5E7EB", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D1D5DB", linewidth = 0.8),
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing.y = unit(4, "mm"),
    plot.margin = margin(12, 14, 12, 12)
  )
save_gg_bundle(
  bubble_plot,
  file.path(output_dir, "phase0_prior_evidence_bubble_matrix"),
  width = 9.8,
  height = max(8.2, min(13.6, 3.6 + 0.18 * nrow(summary_df)))
)

dominant_dim <- aggregate(contribution ~ display_label + segment_label, data = segments_df, sum)
dominant_dim <- dominant_dim[order(dominant_dim$display_label, -dominant_dim$contribution), ]
dominant_dim <- dominant_dim[!duplicated(dominant_dim$display_label), c("display_label", "segment_label")]
lollipop_df <- merge(
  summary_df,
  dominant_dim,
  by = "display_label",
  all.x = TRUE,
  sort = FALSE
)
lollipop_df <- lollipop_df[order(lollipop_df$rank), , drop = FALSE]
lollipop_df$display_label <- factor(lollipop_df$display_label, levels = rev(lollipop_df$display_label))
lollipop_df$segment_label <- factor(lollipop_df$segment_label, levels = dimension_order)
lollipop_df$row_index <- seq_len(nrow(lollipop_df))
retained_rows <- lollipop_df$row_index[lollipop_df$status_label == "Retained"]
excluded_rows <- lollipop_df$row_index[lollipop_df$status_label == "Excluded"]
lolli_band_df <- data.frame(
  xmin = -Inf,
  xmax = Inf,
  ymin = c(
    if (length(retained_rows)) min(retained_rows) - 0.5 else NA_real_,
    if (length(excluded_rows)) min(excluded_rows) - 0.5 else NA_real_
  ),
  ymax = c(
    if (length(retained_rows)) max(retained_rows) + 0.5 else NA_real_,
    if (length(excluded_rows)) max(excluded_rows) + 0.5 else NA_real_
  ),
  band = c("Retained", "Excluded")
)
lolli_band_df <- subset(lolli_band_df, is.finite(ymin) & is.finite(ymax))

lollipop_plot <- ggplot(lollipop_df, aes(y = display_label, x = total_score)) +
  geom_rect(
    data = lolli_band_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = band),
    inherit.aes = FALSE,
    alpha = 0.32,
    color = NA
  ) +
  geom_segment(
    aes(x = 0, xend = total_score, y = display_label, yend = display_label),
    linewidth = 0.75,
    color = "#CBD5E1"
  ) +
  geom_point(
    aes(size = total_score, color = segment_label, shape = status_label),
    stroke = 0.8,
    fill = "white"
  ) +
  scale_fill_manual(values = c("Retained" = "#ECF8F0", "Excluded" = "#F3F4F6"), guide = "none") +
  scale_color_manual(
    values = c(
      "Mechanistic plausibility" = "#F6BD16",
      "Disease specificity" = "#61DDAA",
      "Clinical evidence" = "#5B8FF9",
      "Consistency bonus" = "#9270CA"
    ),
    name = "Dominant evidence"
  ) +
  scale_shape_manual(
    values = c("Retained" = 16, "Excluded" = 1),
    name = "Status"
  ) +
  scale_size_continuous(range = c(2.8, 8.5), name = "Total score") +
  labs(
    title = "Prior Evidence Lollipop",
    subtitle = "Point color marks the dominant evidence source; point size encodes total prior score",
    x = "Prior evidence score",
    y = NULL
  ) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 10.5, hjust = 0.5, color = "#475569"),
    axis.text.y = element_text(size = 8.4, color = "#111827"),
    axis.text.x = element_text(size = 9.5, color = "#374151"),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D1D5DB", linewidth = 0.8),
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing.y = unit(4, "mm"),
    plot.margin = margin(12, 14, 12, 12)
  )
save_gg_bundle(
  lollipop_plot,
  file.path(output_dir, "phase0_prior_evidence_lollipop"),
  width = 9.6,
  height = max(8.2, min(13.6, 3.6 + 0.18 * nrow(summary_df)))
)
