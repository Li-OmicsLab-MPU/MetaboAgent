args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript plot_phase2_phase1_to_phase2_subpanels.R <radar_csv> <contrib_csv> <output_prefix>")
}

suppressPackageStartupMessages({
  lib_path <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
  if (nzchar(lib_path)) {
    .libPaths(unique(c(lib_path, .libPaths())))
  }
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(tidyr)
  library(scales)
})

radar_csv <- args[[1]]
contrib_csv <- args[[2]]
output_prefix <- args[[3]]

radar_df <- read.csv(radar_csv, stringsAsFactors = FALSE, check.names = FALSE)
contrib_df <- read.csv(contrib_csv, stringsAsFactors = FALSE, check.names = FALSE)

if (nrow(radar_df) == 0 || nrow(contrib_df) == 0) {
  stop("Objective-shift subpanel inputs are empty.")
}

metric_levels <- c("AUC", "Biology", "Parsimony", "Independence")
panel_levels <- c("Phase 1 baseline", "Phase 2 winner")

radar_df$metric <- factor(radar_df$metric, levels = metric_levels)
radar_df$panel <- factor(radar_df$panel, levels = panel_levels)

radar_colors <- c("Phase 1 baseline" = "#8AA6C1", "Phase 2 winner" = "#A7493B")
radar_fills <- c("Phase 1 baseline" = alpha("#8AA6C1", 0.18), "Phase 2 winner" = alpha("#A7493B", 0.20))

metric_angles <- tibble(
  metric = factor(metric_levels, levels = metric_levels),
  angle = seq(0, 2 * pi, length.out = length(metric_levels) + 1)[1:length(metric_levels)]
)

radar_plot_df <- radar_df %>%
  left_join(metric_angles, by = "metric") %>%
  arrange(panel, metric) %>%
  group_by(panel) %>%
  mutate(
    x = value * sin(angle),
    y = value * cos(angle)
  ) %>%
  ungroup()

closed_radar_df <- radar_plot_df %>%
  group_by(panel) %>%
  reframe(
    x = c(x, first(x)),
    y = c(y, first(y))
  )

grid_levels <- c(0.25, 0.50, 0.75, 1.00)
grid_df <- bind_rows(lapply(grid_levels, function(r) {
  tibble(
    level = r,
    angle = seq(0, 2 * pi, length.out = 361),
    x = r * sin(angle),
    y = r * cos(angle)
  )
}))

axis_df <- metric_angles %>%
  mutate(
    xend = 1.02 * sin(angle),
    yend = 1.02 * cos(angle),
    label_x = 1.13 * sin(angle),
    label_y = 1.13 * cos(angle),
    metric_label = c("AUC", "Biological\nrelevance", "Cost\nefficiency", "Low\nredundancy")
  )

radar_plot <- ggplot() +
  geom_path(data = grid_df, aes(x = x, y = y, group = level), color = "#E5E7EB", linewidth = 0.4) +
  geom_segment(data = axis_df, aes(x = 0, y = 0, xend = xend, yend = yend), color = "#CBD5E1", linewidth = 0.5) +
  geom_polygon(
    data = closed_radar_df,
    aes(x = x, y = y, group = panel, fill = panel),
    color = NA
  ) +
  geom_path(
    data = closed_radar_df,
    aes(x = x, y = y, group = panel, color = panel),
    linewidth = 1.1
  ) +
  geom_point(
    data = radar_plot_df,
    aes(x = x, y = y, color = panel),
    size = 2.6
  ) +
  geom_text(
    data = axis_df,
    aes(x = label_x, y = label_y, label = metric_label),
    family = "sans",
    size = 3.45,
    fontface = "bold",
    color = "#1F2937",
    lineheight = 0.95
  ) +
  annotate("text", x = 0, y = 1.06, label = "1.0", size = 2.8, color = "#6B7280", family = "sans") +
  annotate("text", x = 0, y = 0.78, label = "0.75", size = 2.8, color = "#9CA3AF", family = "sans") +
  annotate("text", x = 0, y = 0.53, label = "0.50", size = 2.8, color = "#9CA3AF", family = "sans") +
  coord_equal(xlim = c(-1.22, 1.22), ylim = c(-1.18, 1.18), clip = "off") +
  scale_fill_manual(values = radar_fills, name = NULL) +
  scale_color_manual(values = radar_colors, name = NULL) +
  theme_void(base_family = "sans") +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.text = element_text(size = 9.2),
    legend.spacing.x = unit(14, "pt"),
    legend.margin = margin(t = 2, r = 0, b = 0, l = 0),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 18, 4, 18)
  )

contrib_df <- contrib_df %>%
  mutate(
    feature_label = factor(display_name, levels = rev(display_name)),
    contribution_label = ifelse(delta_keep_perf >= 0, "keep helps AUC", "removal helps AUC")
  )

main_track <- ggplot(contrib_df, aes(x = delta_keep_perf, y = feature_label)) +
  geom_vline(xintercept = 0, color = "#CBD5E1", linetype = "22", linewidth = 0.55) +
  geom_segment(
    aes(x = 0, xend = delta_keep_perf, yend = feature_label),
    linewidth = 0.9,
    color = "#94A3B8"
  ) +
  geom_point(
    aes(size = winner_support_score, fill = delta_keep_bio),
    shape = 21,
    color = "#334155",
    stroke = 0.28
  ) +
  scale_fill_gradient2(
    low = "#6FA4C9",
    mid = "#F8FAFC",
    high = "#B4533F",
    midpoint = 0,
    name = expression(Delta * f[bio])
  ) +
  scale_size_continuous(name = "Support", range = c(2.4, 6.0)) +
  labs(x = expression(Delta * "AUC"["keep"]), y = NULL) +
  theme_minimal(base_family = "sans", base_size = 10.2) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 9.2, color = "#111827"),
    axis.title.x = element_text(size = 9.5, face = "bold"),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.just = "left",
    legend.text = element_text(size = 8.8, color = "#111827"),
    legend.title = element_text(size = 9.2, face = "bold"),
    legend.spacing.y = unit(4, "pt"),
    legend.key.width = unit(24, "pt"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 8, 8, 8)
  )

track_long <- contrib_df %>%
  select(
    feature_label,
    direct_prior,
    disease_pathway_align,
    anchor_link,
    coverage_gain
  ) %>%
  pivot_longer(
    cols = -feature_label,
    names_to = "track",
    values_to = "value"
  ) %>%
  mutate(
    track = factor(
      track,
      levels = c("direct_prior", "disease_pathway_align", "anchor_link", "coverage_gain"),
      labels = c("Prior", "Pathway", "Anchor", "Coverage")
    )
  )

track_plot <- ggplot(track_long, aes(x = track, y = feature_label, fill = value)) +
  geom_tile(color = "white", linewidth = 0.35, width = 0.88, height = 0.86) +
  geom_text(aes(label = sprintf("%.2f", value)), family = "sans", size = 2.8, color = "#111827") +
  scale_fill_gradient(
    low = "#F8FAFC",
    high = "#C97C5D",
    name = "Evidence\nscore"
  ) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_family = "sans", base_size = 10) +
  theme(
    panel.grid = element_blank(),
    axis.text.y = element_blank(),
    axis.text.x = element_text(size = 8.8, face = "bold", color = "#374151"),
    legend.position = "bottom",
    legend.title = element_text(size = 9.2, face = "bold"),
    legend.text = element_text(size = 8.8, color = "#111827"),
    legend.key.width = unit(24, "pt"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 8, 8, 8)
  )

panel_title <- function(text) {
  ggdraw() +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    ) +
    draw_label(
      text,
      fontfamily = "sans",
      fontface = "bold",
      size = 13,
      x = 0.5,
      hjust = 0.5,
      color = "#111827"
    )
}

radar_panel <- plot_grid(
  panel_title("Objective Shift"),
  radar_plot,
  ncol = 1,
  rel_heights = c(0.08, 1)
)

main_track_panel <- plot_grid(
  panel_title("Offline leave-one-feature-out influence"),
  main_track,
  ncol = 1,
  rel_heights = c(0.08, 1)
)

track_panel <- plot_grid(
  panel_title("Mechanistic support tracks"),
  track_plot,
  ncol = 1,
  rel_heights = c(0.08, 1)
)

ggsave(paste0(output_prefix, "_a.png"), radar_panel, width = 4.9, height = 7.7, dpi = 320, bg = "white")
ggsave(paste0(output_prefix, "_b.png"), main_track_panel, width = 6.2, height = 7.7, dpi = 320, bg = "white")
ggsave(paste0(output_prefix, "_c.png"), track_panel, width = 4.6, height = 7.7, dpi = 320, bg = "white")
