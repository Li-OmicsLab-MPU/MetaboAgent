args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 6) {
  stop("Usage: Rscript plot_phase1_method_support_previews.R <long_csv> <family_csv> <dot_output_stem> <stacked_output_stem> <barcode_output_stem> <lollipop_output_stem>")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(scales)
})

long_csv <- args[[1]]
family_csv <- args[[2]]
dot_output_stem <- args[[3]]
stacked_output_stem <- args[[4]]
barcode_output_stem <- args[[5]]
lollipop_output_stem <- args[[6]]

wrap_feature_label <- function(x, width = 26) {
  if (!nzchar(x)) return(x)
  x <- gsub(" / ", " /\n", x, fixed = TRUE)
  parts <- strwrap(x, width = width)
  paste(parts, collapse = "\n")
}

long_df <- read.csv(long_csv, check.names = FALSE, stringsAsFactors = FALSE)
family_df <- read.csv(family_csv, check.names = FALSE, stringsAsFactors = FALSE)

feature_levels <- rev(unique(long_df$display_label))
long_df$display_label <- factor(long_df$display_label, levels = feature_levels)
family_df$display_label <- factor(family_df$display_label, levels = feature_levels)

long_df$method_label <- factor(
  long_df$method_label,
  levels = c("Elastic Net", "Lasso", "Random Forest", "LightGBM", "mRMR", "Welch t-test", "FDR/effect-size")
)

family_df$family_label <- factor(
  family_df$family_label,
  levels = c("Regularized", "Tree-based", "Filter/Statistical")
)

family_order <- c("Regularized", "Tree-based", "Filter/Statistical")
family_df$family_label <- factor(family_df$family_label, levels = family_order)

feature_meta <- unique(long_df[, c("feature", "display_name", "display_label", "feature_order")])
feature_meta <- feature_meta[order(feature_meta$feature_order), ]
family_wide <- reshape(
  family_df[, c("display_label", "family_label", "family_support_index")],
  idvar = "display_label",
  timevar = "family_label",
  direction = "wide"
)
names(family_wide) <- sub("^family_support_index\\.", "", names(family_wide))
long_feature_summary <- unique(long_df[, c("display_label", "selection_frequency", "panel_score")])
family_wide <- merge(feature_meta[, c("display_label", "display_name", "feature_order")], family_wide, by = "display_label", all.x = TRUE, sort = FALSE)
family_wide <- merge(family_wide, long_feature_summary, by = "display_label", all.x = TRUE, sort = FALSE)
family_wide <- family_wide[order(family_wide$feature_order), ]

dot_plot <- ggplot(long_df, aes(x = method_label, y = display_label)) +
  geom_point(aes(size = method_frequency, fill = method_frequency), shape = 21, color = "white", stroke = 0.35) +
  scale_size_continuous(range = c(2.4, 10.0), limits = c(0, 1), breaks = c(0.25, 0.50, 0.75, 1.00), labels = label_percent(accuracy = 1)) +
  scale_fill_gradientn(
    colours = c("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"),
    limits = c(0, 1),
    breaks = c(0.25, 0.50, 0.75, 1.00),
    labels = label_percent(accuracy = 1),
    name = "Per-method\nfrequency"
  ) +
  labs(
    title = "Method Support Dot Matrix",
    subtitle = "Final-panel features only",
    x = NULL,
    y = NULL,
    size = "Per-method\nfrequency"
  ) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "#4B5563"),
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, color = "#374151", size = 9.5),
    axis.text.y = element_text(color = "#111827", size = 9.5),
    panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = 0.35),
    panel.grid.major.y = element_line(color = "#F3F4F6", linewidth = 0.30),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 9.5),
    legend.text = element_text(size = 8.5),
    plot.margin = margin(12, 18, 12, 14)
  )

stacked_plot <- ggplot(family_df, aes(x = family_support_index, y = display_label, fill = family_label)) +
  geom_col(width = 0.72, color = "white", linewidth = 0.35) +
  scale_fill_manual(
    values = c(
      "Regularized" = "#4C78A8",
      "Tree-based" = "#54A24B",
      "Filter/Statistical" = "#B279A2"
    )
  ) +
  scale_x_continuous(
    limits = c(0, 3.05),
    breaks = seq(0, 3, 0.5),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Method-Family Stacked Support Profile",
    subtitle = "Segment length = mean selection frequency within each method family",
    x = "Cumulative family support index",
    y = NULL,
    fill = "Method family"
  ) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "#4B5563"),
    axis.text.x = element_text(color = "#374151", size = 9.5),
    axis.text.y = element_text(color = "#111827", size = 9.5),
    panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = 0.35),
    panel.grid.major.y = element_line(color = "#F3F4F6", linewidth = 0.30),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 9.5),
    legend.text = element_text(size = 8.5),
    plot.margin = margin(12, 18, 12, 14)
  )

barcode_plot <- ggplot(long_df, aes(x = method_label, y = display_label)) +
  geom_tile(aes(fill = method_frequency), width = 0.86, height = 0.72, color = "#F3F4F6", linewidth = 0.35) +
  scale_fill_gradientn(
    colours = c("#FFFFFF", "#D9ECF7", "#9ECAE1", "#3182BD", "#08519C"),
    limits = c(0, 1),
    breaks = c(0, 0.25, 0.50, 0.75, 1.00),
    labels = label_percent(accuracy = 1),
    name = "Per-method\nfrequency"
  ) +
  labs(
    title = "Per-feature Method Barcode Plot",
    subtitle = "Compact method-support fingerprint for final-panel features",
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "#4B5563"),
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, color = "#374151", size = 9.5),
    axis.text.y = element_text(color = "#111827", size = 9.5),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 9.5),
    legend.text = element_text(size = 8.5),
    plot.margin = margin(12, 18, 12, 14)
  )

lollipop_df <- family_wide
lollipop_df$panel_anchor <- seq_len(nrow(lollipop_df))
lollipop_long <- reshape(
  lollipop_df[, c("display_label", "panel_anchor", family_order)],
  idvar = c("display_label", "panel_anchor"),
  varying = family_order,
  v.names = "family_support_index",
  timevar = "family_label",
  times = family_order,
  direction = "long"
)
lollipop_long$family_label <- factor(lollipop_long$family_label, levels = family_order)
lollipop_long <- lollipop_long[order(lollipop_long$panel_anchor, lollipop_long$family_label), ]

method_family_lookup <- unique(long_df[, c("method_label", "method_family")])
method_family_lookup$method_family <- factor(method_family_lookup$method_family, levels = family_order)
long_df$method_family <- factor(long_df$method_family, levels = family_order)

lollipop_plot <- ggplot() +
  geom_segment(
    data = lollipop_df,
    aes(x = 0, xend = selection_frequency, y = display_label, yend = display_label),
    color = "#D1D5DB",
    linewidth = 0.7
  ) +
  geom_point(
    data = lollipop_df,
    aes(x = selection_frequency, y = display_label, size = panel_score),
    shape = 21,
    fill = "#1F77B4",
    color = "white",
    stroke = 0.35
  ) +
  geom_point(
    data = long_df,
    aes(
      x = 1.06 + (as.numeric(method_label) - 1) * 0.055,
      y = display_label,
      fill = method_frequency
    ),
    shape = 22,
    size = 2.3,
    color = "#E5E7EB",
    stroke = 0.25
  ) +
  scale_size_continuous(range = c(2.5, 8.5), guide = "none") +
  scale_fill_gradientn(
    colours = c("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"),
    limits = c(0, 1),
    breaks = c(0.25, 0.50, 0.75, 1.00),
    labels = label_percent(accuracy = 1),
    name = "Method support"
  ) +
  scale_x_continuous(
    limits = c(0, 1.48),
    breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
    sec.axis = dup_axis(
      breaks = 1.06 + (seq_along(levels(long_df$method_label)) - 1) * 0.055,
      labels = levels(long_df$method_label),
      name = NULL
    )
  ) +
  labs(
    title = "Consensus Decomposition Lollipop",
    subtitle = "Lollipop = selection frequency; right-side tiles = per-method support",
    x = "Selection frequency",
    y = NULL
  ) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "#4B5563"),
    axis.text.x = element_text(color = "#374151", size = 9.5),
    axis.text.y = element_text(color = "#111827", size = 9.5),
    axis.text.x.top = element_text(angle = 35, hjust = 0, vjust = 0.2, color = "#374151", size = 8.5),
    panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = 0.35),
    panel.grid.major.y = element_line(color = "#F3F4F6", linewidth = 0.30),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 9.5),
    legend.text = element_text(size = 8.5),
    plot.margin = margin(12, 20, 12, 14)
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

save_plot(dot_plot, dot_output_stem, width = 10.6, height = 7.8)
save_plot(stacked_plot, stacked_output_stem, width = 10.6, height = 7.8)
save_plot(barcode_plot, barcode_output_stem, width = 10.6, height = 7.8)
save_plot(lollipop_plot, lollipop_output_stem, width = 12.2, height = 7.8)
