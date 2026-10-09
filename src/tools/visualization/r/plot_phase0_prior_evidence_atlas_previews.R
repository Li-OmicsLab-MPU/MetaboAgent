args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Usage: Rscript plot_phase0_prior_evidence_atlas_previews.R <summary_csv> <segments_csv> <output_dir>")
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

summary_csv <- args[[1]]
segments_csv <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

summary_df <- read.csv(summary_csv, check.names = FALSE, stringsAsFactors = FALSE)
segments_df <- read.csv(segments_csv, check.names = FALSE, stringsAsFactors = FALSE)

wrap_label <- function(x, width = 22) {
  if (!nzchar(x)) return(x)
  paste(strwrap(x, width = width), collapse = "\n")
}

save_plot_bundle <- function(plot_obj, output_stem, width, height) {
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
summary_df$wrapped_label <- vapply(summary_df$display_label, wrap_label, character(1))
summary_df$status_label <- factor(summary_df$status_label, levels = c("Retained", "Excluded"))
summary_df$row_id <- rev(seq_len(nrow(summary_df)))

dimension_order <- c(
  "Mechanistic plausibility",
  "Disease specificity",
  "Clinical evidence",
  "Consistency bonus"
)
dimension_colors <- c(
  "Mechanistic plausibility" = "#E7B416",
  "Disease specificity" = "#5AC8A5",
  "Clinical evidence" = "#5B8FF9",
  "Consistency bonus" = "#9270CA"
)
status_colors <- c("Retained" = "#DFF2E5", "Excluded" = "#EEF1F4")
status_line_colors <- c("Retained" = "#1F7A3E", "Excluded" = "#6B7280")

segments_df <- merge(
  segments_df,
  summary_df[, c("display_label", "wrapped_label", "row_id", "rank", "status_label", "total_score", "pubmed_hit_count")],
  by = "display_label",
  all.x = TRUE,
  sort = FALSE
)
if ("rank.x" %in% names(segments_df)) {
  segments_df$rank <- segments_df$rank.x
}
if ("status_label.x" %in% names(segments_df)) {
  segments_df$status_label <- segments_df$status_label.x
}
if ("total_score.x" %in% names(segments_df)) {
  segments_df$total_score <- segments_df$total_score.x
}
segments_df$segment_label <- factor(segments_df$segment_label, levels = dimension_order)
segments_df$segment_short <- factor(
  c(
    "Mechanistic\nplausibility",
    "Disease\nspecificity",
    "Clinical\nevidence",
    "Consistency\nbonus"
  )[match(segments_df$segment_label, dimension_order)],
  levels = c("Mechanistic\nplausibility", "Disease\nspecificity", "Clinical\nevidence", "Consistency\nbonus")
)

threshold_numeric <- 5.49
if ("total_score" %in% names(summary_df)) {
  threshold_numeric <- 5.49
}
if ("status_label" %in% names(summary_df) && any(summary_df$status_label == "Retained") && any(summary_df$status_label == "Excluded")) {
  threshold_numeric <- mean(
    c(
      min(summary_df$total_score[summary_df$status_label == "Retained"], na.rm = TRUE),
      max(summary_df$total_score[summary_df$status_label == "Excluded"], na.rm = TRUE)
    )
  )
}

## Atlas preview --------------------------------------------------------------
retained_rows <- summary_df$row_id[summary_df$status_label == "Retained"]
excluded_rows <- summary_df$row_id[summary_df$status_label == "Excluded"]
retained_band <- data.frame(
  ymin = if (length(retained_rows)) min(retained_rows) - 0.5 else NA_real_,
  ymax = if (length(retained_rows)) max(retained_rows) + 0.5 else NA_real_
)
excluded_band <- data.frame(
  ymin = if (length(excluded_rows)) min(excluded_rows) - 0.5 else NA_real_,
  ymax = if (length(excluded_rows)) max(excluded_rows) + 0.5 else NA_real_
)
separator_y <- if (length(retained_rows) && length(excluded_rows)) {
  (min(retained_rows) + max(excluded_rows)) / 2
} else {
  NA_real_
}
separator_band <- data.frame(
  ymin = separator_y - 0.48,
  ymax = separator_y + 0.48
)
label_df_retained <- subset(summary_df, status_label == "Retained")
label_df_excluded <- subset(summary_df, status_label == "Excluded")

label_panel <- ggplot() +
  geom_text(
    data = label_df_retained,
    aes(x = 0.98, y = row_id, label = wrapped_label),
    hjust = 1,
    size = 8.3 / .pt,
    fontface = "bold",
    color = "#111827",
    lineheight = 0.95
  ) +
  geom_text(
    data = label_df_excluded,
    aes(x = 0.98, y = row_id, label = wrapped_label),
    hjust = 1,
    size = 8.3 / .pt,
    color = "#374151",
    lineheight = 0.95
  ) +
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(min(summary_df$row_id) - 0.5, max(summary_df$row_id) + 0.7),
    clip = "off"
  ) +
  theme_void(base_family = "sans") +
  theme(
    plot.margin = margin(4, 6, 4, 4)
  )

total_panel <- ggplot(summary_df, aes(y = row_id, x = total_score)) +
  geom_rect(
    data = retained_band,
    aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = "#EAF6ED",
    alpha = 0.95,
    color = NA
  ) +
  geom_rect(
    data = excluded_band,
    aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = "#F5F7F9",
    alpha = 0.98,
    color = NA
  ) +
  geom_segment(aes(x = 0, xend = total_score, yend = row_id), linewidth = 0.72, color = "#CBD5E1") +
  geom_point(
    aes(fill = status_label),
    shape = 21,
    size = 2.9,
    stroke = 0.55,
    color = "#334155"
  ) +
  geom_text(
    aes(label = sprintf("%.1f", total_score), x = total_score + 0.16),
    size = 2.85,
    hjust = 0,
    color = "#334155"
  ) +
  geom_vline(xintercept = threshold_numeric, color = "#B45309", linewidth = 0.9, linetype = "22") +
  annotate(
    "label",
    x = threshold_numeric + 0.14,
    y = max(summary_df$row_id) - 0.1,
    label = sprintf("Adaptive threshold = %.2f", threshold_numeric),
    size = 3.0,
    color = "#92400E",
    fill = "#FFF6E7",
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c(
      "Retained" = "#1F7A3E",
      "Excluded" = "white"
    ),
    guide = "none"
  ) +
  coord_cartesian(
    xlim = c(0, max(summary_df$total_score, na.rm = TRUE) + 1.2),
    ylim = c(min(summary_df$row_id) - 0.5, max(summary_df$row_id) + 0.7),
    clip = "off"
  ) +
  labs(x = "Total prior score", y = NULL) +
  theme_bw(base_family = "sans", base_size = 10.5) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = 9.0, color = "#374151"),
    axis.title.x = element_text(size = 11, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D1D5DB", linewidth = 0.8),
    plot.margin = margin(4, 4, 4, 4)
  )

spark_base_theme <- theme_bw(base_family = "sans", base_size = 10) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title = element_blank(),
    panel.grid.major.x = element_line(color = "#EDF2F7", linewidth = 0.28),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D1D5DB", linewidth = 0.6),
    plot.title = element_text(size = 8.1, face = "bold", hjust = 0.5, color = "#334155"),
    axis.text.x = element_text(size = 7.3, color = "#475569"),
    plot.margin = margin(4, 3, 4, 3)
  )

make_spark_track <- function(track_name) {
  df <- subset(segments_df, segment_label == track_name)
  x_min <- min(0, min(df$contribution, na.rm = TRUE))
  x_max <- max(df$contribution, na.rm = TRUE)
  ggplot() +
    geom_rect(
      data = retained_band,
      aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,
      fill = "#EAF6ED",
      alpha = 0.95,
      color = NA
    ) +
    geom_rect(
      data = excluded_band,
      aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,
      fill = "#F5F7F9",
      alpha = 0.98,
      color = NA
    ) +
    geom_vline(xintercept = 0, color = "#D1D5DB", linewidth = 0.4) +
    geom_rect(
      data = df,
      aes(
        xmin = pmin(0, contribution),
        xmax = pmax(0, contribution),
        ymin = row_id - 0.28,
        ymax = row_id + 0.28
      ),
      inherit.aes = FALSE,
      fill = dimension_colors[[track_name]],
      alpha = 0.92,
      color = NA
    ) +
    geom_text(
      data = df,
      aes(x = contribution + ifelse(contribution >= 0, 0.12, -0.12), y = row_id, label = sprintf("%.1f", contribution)),
      hjust = ifelse(df$contribution >= 0, 0, 1),
      size = 2.35,
      color = "#334155"
    ) +
    coord_cartesian(
      xlim = c(x_min - 0.15, x_max + 0.72),
      ylim = c(min(summary_df$row_id) - 0.5, max(summary_df$row_id) + 0.7),
      clip = "off"
    ) +
    labs(title = track_name) +
    spark_base_theme
}

mech_track <- make_spark_track("Mechanistic plausibility")
spec_track <- make_spark_track("Disease specificity")
clin_track <- make_spark_track("Clinical evidence")
cons_track <- make_spark_track("Consistency bonus")

atlas_top_title <- ggdraw() +
  draw_label("Prior Evidence Atlas", fontface = "bold", size = 17, x = 0.5, hjust = 0.5)

atlas_body <- plot_grid(
  label_panel,
  total_panel,
  mech_track,
  spec_track,
  clin_track,
  cons_track,
  nrow = 1,
  align = "h",
  axis = "tb",
  rel_widths = c(1.85, 4.45, 1.3, 1.3, 1.22, 1.14)
)

atlas_legend <- ggdraw() +
  draw_label("Evidence Contributions", x = 0.08, y = 0.88, hjust = 0, fontface = "bold", size = 9.6) +
  draw_line(x = c(0.08, 0.92), y = c(0.71, 0.71), color = "#E5E7EB", linewidth = 0.8) +
  draw_label("Mech = yellow  |  Spec = green  |  Clin = blue  |  Cons = purple", x = 0.28, y = 0.45, size = 8.6, color = "#475569") +
  draw_label("Retained = pale green background  |  Excluded = pale gray background  |  Adaptive threshold = dashed vertical line", x = 0.77, y = 0.45, size = 8.05, color = "#475569")

atlas_plot <- plot_grid(
  atlas_top_title,
  atlas_body,
  atlas_legend,
  ncol = 1,
  rel_heights = c(0.06, 1, 0.1)
)

atlas_height <- max(8.6, min(13.8, 4.2 + 0.19 * nrow(summary_df)))
save_plot_bundle(
  atlas_plot,
  file.path(output_dir, "phase0_prior_evidence_atlas"),
  width = 15.0,
  height = atlas_height
)

## Ranked evidence track plot -----------------------------------------------
wide_df <- reshape(
  segments_df[, c("rank", "segment_label", "contribution")],
  idvar = "rank",
  timevar = "segment_label",
  direction = "wide"
)
colnames(wide_df) <- sub("^contribution\\.", "", colnames(wide_df))
wide_df <- merge(
  summary_df[, c("rank", "wrapped_label", "total_score", "status_label")],
  wide_df,
  by = "rank",
  all.x = TRUE,
  sort = FALSE
)
wide_df <- wide_df[order(wide_df$rank), , drop = FALSE]
for (nm in dimension_order) {
  if (!(nm %in% names(wide_df))) wide_df[[nm]] <- 0
}

short_label <- function(x, width = 12) {
  x <- gsub("\n", " ", x, fixed = TRUE)
  if (nchar(x) <= width) return(x)
  paste0(substr(x, 1, width - 3), "...")
}

wide_df$short_label <- vapply(wide_df$wrapped_label, short_label, character(1))
wide_df$index_label <- paste0(wide_df$rank, " ", wide_df$short_label)

column_bands <- data.frame(
  xmin = seq_len(nrow(wide_df)) - 0.5,
  xmax = seq_len(nrow(wide_df)) + 0.5,
  band = ifelse(seq_len(nrow(wide_df)) %% 2 == 1, "odd", "even")
)

retained_labels_df <- subset(wide_df, status_label == "Retained")
retained_labels_df$label_y <- retained_labels_df$total_score + ifelse(retained_labels_df$rank %% 2 == 1, 0.55, 0.9)

signal_long <- rbind(
  data.frame(rank = wide_df$rank, track = "Mechanistic plausibility", value = wide_df[["Mechanistic plausibility"]]),
  data.frame(rank = wide_df$rank, track = "Disease specificity", value = wide_df[["Disease specificity"]]),
  data.frame(rank = wide_df$rank, track = "Clinical evidence", value = wide_df[["Clinical evidence"]]),
  data.frame(rank = wide_df$rank, track = "Consistency bonus", value = wide_df[["Consistency bonus"]])
)
signal_long$track <- factor(signal_long$track, levels = c(dimension_order, "Metabolite"))

label_track_df <- wide_df
label_track_df$track <- factor("Metabolite", levels = c(dimension_order, "Metabolite"))
label_track_df$label_y <- 0.5

top_panel <- ggplot(wide_df, aes(x = rank, y = total_score)) +
  geom_rect(
    data = column_bands,
    aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = band),
    inherit.aes = FALSE,
    alpha = 0.22,
    color = NA
  ) +
  geom_rect(
    data = data.frame(
      xmin = c(0.5, 0.5),
      xmax = c(nrow(wide_df) + 0.5, nrow(wide_df) + 0.5),
      ymin = c(threshold_numeric, -Inf),
      ymax = c(Inf, threshold_numeric),
      band = c("Retained", "Excluded")
    ),
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = band),
    inherit.aes = FALSE,
    alpha = 0.26,
    color = NA
  ) +
  geom_vline(xintercept = wide_df$rank, color = "#E5E7EB", linewidth = 0.22) +
  geom_line(color = "#94A3B8", linewidth = 0.65) +
  geom_point(aes(fill = status_label), shape = 21, size = 2.6, stroke = 0.45, color = "#334155") +
  geom_text(
    data = retained_labels_df,
    aes(x = rank, y = label_y, label = short_label),
    inherit.aes = FALSE,
    angle = 55,
    hjust = 0,
    vjust = 0,
    size = 2.45,
    color = "#1F2937"
  ) +
  geom_hline(yintercept = threshold_numeric, color = "#B45309", linetype = "22", linewidth = 0.82) +
  annotate(
    "label",
    x = nrow(wide_df) * 0.86,
    y = threshold_numeric + 0.22,
    label = sprintf("Adaptive threshold %.2f", threshold_numeric),
    size = 3.0,
    fill = "#FFF6E7",
    color = "#92400E",
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c("odd" = "#F8FAFC", "even" = "#FFFFFF", "Retained" = "#1F7A3E", "Excluded" = "white"),
    breaks = NULL,
    guide = "none"
  ) +
  labs(title = "Ranked Evidence Track Plot", y = "Total prior score", x = NULL) +
  theme_bw(base_family = "sans", base_size = 10.5) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 8.8, color = "#374151"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D1D5DB", linewidth = 0.8),
    plot.margin = margin(8, 8, 4, 8)
  )

track_panel <- ggplot(signal_long, aes(x = rank, y = value, fill = track)) +
  geom_rect(
    data = column_bands,
    aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = band),
    inherit.aes = FALSE,
    alpha = 0.22,
    color = NA
  ) +
  geom_col(width = 0.82, show.legend = FALSE) +
  geom_tile(
    data = label_track_df,
    aes(x = rank, y = label_y, fill = status_label),
    inherit.aes = FALSE,
    height = 0.78,
    width = 0.92,
    color = "white",
    linewidth = 0.18
  ) +
  geom_text(
    data = label_track_df,
    aes(x = rank, y = label_y, label = index_label),
    inherit.aes = FALSE,
    angle = 90,
    hjust = 1,
    vjust = 0.5,
    size = 2.05,
    color = "#334155"
  ) +
  facet_grid(rows = vars(track), scales = "free_y", switch = "y") +
  geom_vline(xintercept = wide_df$rank, color = "#E5E7EB", linewidth = 0.2) +
  scale_fill_manual(
    values = c(
      "odd" = "#F8FAFC",
      "even" = "#FFFFFF",
      dimension_colors,
      "Retained" = "#B7E1C0",
      "Excluded" = "#DDE3EA"
    ),
    breaks = NULL
  ) +
  labs(x = NULL, y = NULL) +
  theme_bw(base_family = "sans", base_size = 10) +
  theme(
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 0, size = 8.8, face = "bold", color = "#334155"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 7.8, color = "#374151"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#EEF2F7", linewidth = 0.28),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D1D5DB", linewidth = 0.55),
    strip.background = element_rect(fill = "#F8FAFC", color = "#D1D5DB"),
    plot.margin = margin(0, 8, 2, 8)
  )

track_subtitle <- ggdraw() +
  draw_label(
    "Genome-browser-inspired multi-track view: metabolite names are integrated as an aligned bottom track sharing the exact same x positions as the four evidence dimensions",
    size = 10.0,
    color = "#475569",
    x = 0.5,
    y = 0.56,
    hjust = 0.5
  )

track_plot <- plot_grid(
  track_subtitle,
  top_panel,
  track_panel,
  ncol = 1,
  rel_heights = c(0.08, 0.28, 0.74)
)

track_height <- max(8.8, min(12.8, 7.6 + 0.02 * nrow(summary_df)))
save_plot_bundle(
  track_plot,
  file.path(output_dir, "phase0_ranked_evidence_track_plot"),
  width = 12.6,
  height = track_height
)
