args <- commandArgs(trailingOnly = TRUE)

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

if (length(args) < 5) {
  stop("Usage: Rscript plot_phase0_prior_evidence_atlas.R <summary_csv> <segments_csv> <output_stem> <title> <threshold_value> [theme_json]")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}
source(file.path(get_script_dir(), "font_family_helper.R"))
font_family <- resolve_font_family(NULL)

suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
})

summary_csv <- args[[1]]
segments_csv <- args[[2]]
output_stem <- args[[3]]
plot_title <- args[[4]]
threshold_numeric <- as.numeric(args[[5]])
theme_json <- ifelse(length(args) >= 6, args[[6]], "")
# Keep the font family from the theme so Nature style can stay consistent.

text_color <- "#1F2937"
axis_text_color <- "#374151"
grid_color <- "#E5E7EB"
panel_border_color <- "#D1D5DB"
reference_color <- "#BBBDC0"
highlight_color <- "#202754"
positive_color <- "#19AA9D"
neutral_color <- "#BBBDC0"
uncertainty_fill <- "#F2F4F7"
retained_band_fill <- "#EAF6ED"
excluded_band_fill <- "#F5F7F9"
spark_grid_color <- "#EDF2F7"
spark_text_color <- axis_text_color
dimension_colors <- c(
  "Mechanistic plausibility" = "#19AA9D",
  "Disease specificity" = "#3D85C6",
  "Clinical evidence" = "#E96362",
  "Consistency bonus" = "#9B7FC4"
)

if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  text_color <- theme$text_color
  axis_text_color <- theme$axis_text_color
  grid_color <- theme$grid_color
  panel_border_color <- theme$panel_border_color
  reference_color <- theme$semantic_colors$reference
  highlight_color <- theme$semantic_colors$highlight
  positive_color <- theme$semantic_colors$positive
  neutral_color <- theme$semantic_colors$neutral
  uncertainty_fill <- theme$primary_fill
  retained_band_fill <- if (!is.null(theme$primary_fill) && nzchar(theme$primary_fill)) {
    theme$primary_fill
  } else {
    uncertainty_fill
  }
  excluded_band_fill <- if (!is.null(theme$background_color) && nzchar(theme$background_color)) {
    theme$background_color
  } else {
    "#F8FAFC"
  }
  spark_grid_color <- if (!is.null(theme$grid_color) && nzchar(theme$grid_color)) theme$grid_color else grid_color
  spark_text_color <- axis_text_color
  dimension_colors <- c(
    "Mechanistic plausibility" = theme$semantic_colors$mechanistic,
    "Disease specificity" = theme$semantic_colors$phase1,
    "Clinical evidence" = theme$semantic_colors$clinical,
    "Consistency bonus" = theme$semantic_colors$redundancy
  )
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- resolve_font_family(theme$font_family)
  }
}

summary_df <- read.csv(summary_csv, check.names = FALSE, stringsAsFactors = FALSE)
segments_df <- read.csv(segments_csv, check.names = FALSE, stringsAsFactors = FALSE)

wrap_label <- function(x, width = 22) {
  if (!nzchar(x)) return(x)
  paste(strwrap(x, width = width), collapse = "\n")
}

save_plot_bundle <- function(plot_obj, output_stem, width, height) {
  grDevices::cairo_pdf(
    paste0(output_stem, ".pdf"),
    width = width,
    height = height,
    family = "sans"
  )
  print(plot_obj)
  dev.off()

  if (requireNamespace("ragg", quietly = TRUE)) {
    ragg::agg_png(
      paste0(output_stem, ".png"),
      width = width,
      height = height,
      units = "in",
      res = 600
    )
  } else {
    png(
      paste0(output_stem, ".png"),
      width = width * 600,
      height = height * 600,
      res = 600,
      type = "cairo"
    )
  }
  print(plot_obj)
  dev.off()

  if (requireNamespace("svglite", quietly = TRUE)) {
    svglite::svglite(paste0(output_stem, ".svg"), width = width, height = height)
  } else {
    grDevices::svg(paste0(output_stem, ".svg"), width = width, height = height)
  }
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

segments_df <- merge(
  segments_df,
  summary_df[, c("display_label", "wrapped_label", "row_id", "rank", "status_label", "total_score")],
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
label_df_retained <- subset(summary_df, status_label == "Retained")
label_df_excluded <- subset(summary_df, status_label == "Excluded")

label_panel <- ggplot() +
  geom_text(
    data = label_df_retained,
    aes(x = 0.98, y = row_id, label = wrapped_label),
    hjust = 1,
    size = 8.3 / .pt,
    fontface = "bold",
    color = text_color,
    lineheight = 0.95
  ) +
  geom_text(
    data = label_df_excluded,
    aes(x = 0.98, y = row_id, label = wrapped_label),
    hjust = 1,
    size = 8.3 / .pt,
    color = axis_text_color,
    lineheight = 0.95
  ) +
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(min(summary_df$row_id) - 0.5, max(summary_df$row_id) + 0.7),
    clip = "off"
  ) +
  theme_void(base_family = font_family) +
  theme(
    plot.margin = margin(4, 6, 4, 4)
  )

total_panel <- ggplot(summary_df, aes(y = row_id, x = total_score)) +
  geom_rect(
    data = retained_band,
    aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = retained_band_fill,
    alpha = 0.95,
    color = NA
  ) +
  geom_rect(
    data = excluded_band,
    aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = excluded_band_fill,
    alpha = 0.98,
    color = NA
  ) +
  geom_segment(aes(x = 0, xend = total_score, yend = row_id), linewidth = 0.72, color = reference_color) +
  geom_point(
    aes(fill = status_label),
    shape = 21,
    size = 2.9,
    stroke = 0.55,
    color = text_color
  ) +
  geom_text(
    aes(label = sprintf("%.1f", total_score), x = total_score + 0.16),
    size = 2.85,
    hjust = 0,
    color = text_color
  ) +
  geom_vline(xintercept = threshold_numeric, color = highlight_color, linewidth = 0.9, linetype = "22") +
  annotate(
    "label",
    x = threshold_numeric + 0.14,
    y = max(summary_df$row_id) - 0.1,
    label = sprintf("Adaptive threshold = %.2f", threshold_numeric),
    size = 3.0,
    color = highlight_color,
    fill = uncertainty_fill,
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c(
      "Retained" = positive_color,
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
  theme_bw(base_family = font_family, base_size = 10.5) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = 9.0, color = axis_text_color),
    axis.title.x = element_text(size = 11, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = grid_color, linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = panel_border_color, linewidth = 0.8),
    plot.margin = margin(4, 4, 4, 4)
  )

spark_base_theme <- theme_bw(base_family = font_family, base_size = 10) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title = element_blank(),
    panel.grid.major.x = element_line(color = spark_grid_color, linewidth = 0.28),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = panel_border_color, linewidth = 0.6),
    plot.title = element_text(size = 8.1, face = "bold", hjust = 0.5, color = text_color),
    axis.text.x = element_text(size = 7.3, color = spark_text_color),
    plot.margin = margin(4, 3, 4, 3)
  )

make_spark_track <- function(track_name) {
  df <- subset(segments_df, segment_label == track_name)
  x_min <- min(0, min(df$contribution, na.rm = TRUE))
  x_max <- max(df$contribution, na.rm = TRUE)
  if (!is.finite(x_min)) x_min <- 0
  if (!is.finite(x_max)) x_max <- 1
  if (track_name == "Consistency bonus") {
    x_min <- min(-1, x_min)
    x_max <- max(1, x_max)
  }
  if (x_max <= x_min) {
    x_max <- x_min + 1
  }
  ggplot() +
    geom_rect(
      data = retained_band,
      aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,
      fill = retained_band_fill,
      alpha = 0.95,
      color = NA
    ) +
    geom_rect(
      data = excluded_band,
      aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,
      fill = excluded_band_fill,
      alpha = 0.98,
      color = NA
    ) +
    geom_vline(xintercept = 0, color = panel_border_color, linewidth = 0.4) +
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
      color = text_color
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
  draw_label(plot_title, fontface = "bold", size = 17, x = 0.5, y = 0.5, hjust = 0.5, vjust = 0.5)

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
  draw_label("Evidence Contributions", x = 0.08, y = 0.88, hjust = 0, fontface = "bold", size = 9.6, color = text_color) +
  draw_line(x = c(0.08, 0.92), y = c(0.71, 0.71), color = grid_color, linewidth = 0.8) +
  draw_label("Mech = mechanistic  |  Spec = specificity  |  Clin = clinical  |  Cons = consistency", x = 0.28, y = 0.45, size = 8.6, color = axis_text_color) +
  draw_label("Retained = themed positive tint  |  Excluded = themed neutral tint  |  Adaptive threshold = dashed vertical line", x = 0.77, y = 0.45, size = 8.05, color = axis_text_color)

atlas_plot <- plot_grid(
  atlas_top_title,
  atlas_body,
  atlas_legend,
  ncol = 1,
  rel_heights = c(0.06, 1, 0.1)
)

atlas_height <- max(8.6, min(13.8, 4.2 + 0.19 * nrow(summary_df)))
save_plot_bundle(atlas_plot, output_stem, width = 15.0, height = atlas_height)
