args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript plot_phase2_phase1_to_phase2_summary.R <radar_csv> <contrib_csv> <output_prefix> [theme_json]")
}

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

source(file.path(get_script_dir(), "font_family_helper.R"))

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
theme_json <- ifelse(length(args) >= 4, args[[4]], "")
font_family <- resolve_font_family(NULL)
text_color <- "#1F2937"
axis_text_color <- "#374151"
grid_color <- "#E5E7EB"
panel_border_color <- "#D7DEE8"
baseline_color <- "#6D9185"
winner_color <- "#6E3B33"
positive_color <- "#6D9185"
negative_color <- "#B46A5B"
neutral_color <- "#94A3B8"
if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- resolve_font_family(theme$font_family)
  }
  if (!is.null(theme$text_color) && nzchar(theme$text_color)) text_color <- theme$text_color
  if (!is.null(theme$axis_text_color) && nzchar(theme$axis_text_color)) axis_text_color <- theme$axis_text_color
  if (!is.null(theme$grid_color) && nzchar(theme$grid_color)) grid_color <- theme$grid_color
  if (!is.null(theme$panel_border_color) && nzchar(theme$panel_border_color)) panel_border_color <- theme$panel_border_color
  if (!is.null(theme$semantic_colors$baseline) && nzchar(theme$semantic_colors$baseline)) baseline_color <- theme$semantic_colors$baseline
  if (!is.null(theme$semantic_colors$winner) && nzchar(theme$semantic_colors$winner)) winner_color <- theme$semantic_colors$winner
  if (!is.null(theme$semantic_colors$positive) && nzchar(theme$semantic_colors$positive)) positive_color <- theme$semantic_colors$positive
  if (!is.null(theme$semantic_colors$negative) && nzchar(theme$semantic_colors$negative)) negative_color <- theme$semantic_colors$negative
  if (!is.null(theme$semantic_colors$reference) && nzchar(theme$semantic_colors$reference)) neutral_color <- theme$semantic_colors$reference
}
font_family <- resolve_font_family(font_family)

radar_df <- read.csv(radar_csv, stringsAsFactors = FALSE, check.names = FALSE)
contrib_df <- read.csv(contrib_csv, stringsAsFactors = FALSE, check.names = FALSE)

if (nrow(radar_df) == 0 || nrow(contrib_df) == 0) {
  stop("Figure F inputs are empty.")
}

metric_levels <- c("AUC", "Biology", "Parsimony", "Independence")
panel_levels <- c("Phase 1 baseline", "Phase 2 winner")

radar_df$metric <- factor(radar_df$metric, levels = metric_levels)
radar_df$panel <- factor(radar_df$panel, levels = panel_levels)

radar_colors <- c("Phase 1 baseline" = baseline_color, "Phase 2 winner" = winner_color)
radar_fills <- c("Phase 1 baseline" = alpha(baseline_color, 0.18), "Phase 2 winner" = alpha(winner_color, 0.20))

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
  geom_path(data = grid_df, aes(x = x, y = y, group = level), color = grid_color, linewidth = 0.4) +
  geom_segment(data = axis_df, aes(x = 0, y = 0, xend = xend, yend = yend), color = panel_border_color, linewidth = 0.5) +
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
    family = font_family,
    size = 3.45,
    fontface = "bold",
    color = "#1F2937",
    lineheight = 0.95
  ) +
  annotate("text", x = 0, y = 1.06, label = "1.0", size = 2.8, color = "#6B7280", family = font_family) +
  annotate("text", x = 0, y = 0.78, label = "0.75", size = 2.8, color = "#9CA3AF", family = font_family) +
  annotate("text", x = 0, y = 0.53, label = "0.50", size = 2.8, color = "#9CA3AF", family = font_family) +
  coord_equal(xlim = c(-1.22, 1.22), ylim = c(-1.18, 1.18), clip = "off") +
  scale_fill_manual(values = radar_fills, name = NULL) +
  scale_color_manual(values = radar_colors, name = NULL) +
  theme_void(base_family = font_family) +
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
    display_name_unique = make.unique(as.character(display_name), sep = " | "),
    feature_label = factor(display_name_unique, levels = rev(display_name_unique)),
    contribution_label = ifelse(delta_keep_perf >= 0, "keep helps AUC", "removal helps AUC")
  )

feature_levels <- levels(contrib_df$feature_label)
feature_axis_scale <- scale_y_discrete(
  limits = feature_levels,
  drop = FALSE,
  expand = expansion(mult = c(0.02, 0.02))
)

main_track <- ggplot(contrib_df, aes(x = delta_keep_perf, y = feature_label)) +
  geom_vline(xintercept = 0, color = neutral_color, linetype = "22", linewidth = 0.55) +
  geom_segment(
    aes(x = 0, xend = delta_keep_perf, yend = feature_label),
    linewidth = 0.9,
    color = winner_color
  ) +
  geom_point(
    aes(size = winner_support_score, fill = delta_keep_bio),
    shape = 21,
    color = "#334155",
    stroke = 0.28
  ) +
  scale_fill_gradient2(
    low = negative_color,
    mid = "#F8FAFC",
    high = positive_color,
    midpoint = 0,
    name = expression(Delta * f[bio])
  ) +
  scale_size_continuous(name = "Support", range = c(2.4, 6.0)) +
  feature_axis_scale +
  labs(x = expression(Delta * "AUC"["keep"]), y = NULL) +
  theme_minimal(base_family = font_family, base_size = 10.2) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 9.2, color = text_color),
    axis.title.x = element_text(size = 9.5, face = "bold", color = text_color),
    axis.text.x = element_text(size = 8.8, color = axis_text_color),
    panel.grid.major.x = element_line(color = grid_color, linewidth = 0.28),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.text = element_text(size = 8.8, color = text_color),
    legend.title = element_text(size = 9.2, face = "bold", color = text_color),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(color = panel_border_color, fill = NA, linewidth = 0.6),
    plot.margin = margin(0, 2, 4, 8)
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
  geom_text(aes(label = sprintf("%.2f", value)), family = font_family, size = 2.8, color = text_color) +
  scale_fill_gradient(
    low = "#F8FAFC",
    high = winner_color,
    name = "Evidence\nscore"
  ) +
  feature_axis_scale +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_family = font_family, base_size = 10) +
  theme(
    panel.grid = element_blank(),
    axis.text.y = element_blank(),
    axis.text.x = element_text(size = 8.8, face = "bold", color = axis_text_color),
    legend.position = "bottom",
    legend.title = element_text(size = 9.2, face = "bold", color = text_color),
    legend.text = element_text(size = 8.8, color = text_color),
    legend.key.width = unit(24, "pt"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(color = panel_border_color, fill = NA, linewidth = 0.6),
    plot.margin = margin(0, 8, 4, 0)
  )

panel_title <- function(text) {
  ggdraw() +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    ) +
    draw_label(
      text,
      fontfamily = font_family,
      fontface = "bold",
      size = 13,
      x = 0.5,
      hjust = 0.5,
      color = text_color
    )
}

radar_panel <- plot_grid(
  panel_title("Objective profile shift"),
  radar_plot,
  ncol = 1,
  rel_heights = c(0.08, 1)
)

main_track_panel <- plot_grid(
  panel_title("Winner-Feature Contribution"),
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

contrib_panel <- plot_grid(
  main_track_panel,
  track_panel,
  ncol = 2,
  rel_widths = c(2.05, 1.35),
  align = "v",
  axis = "lr"
)

title_plot <- ggdraw() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  draw_label(
    "Objective Shift and Feature Contribution of the Winner Panel",
    fontfamily = font_family,
    fontface = "bold",
    size = 14,
    x = 0.5,
    hjust = 0.5,
    color = "#111827"
  )

final_plot <- plot_grid(
  title_plot,
  plot_grid(radar_panel, contrib_panel, ncol = 2, rel_widths = c(1.00, 1.90), align = "h", axis = "t"),
  ncol = 1,
  rel_heights = c(0.08, 1)
)

ggsave(paste0(output_prefix, ".pdf"), final_plot, width = 14.2, height = 8.1, device = cairo_pdf, bg = "white")
ggsave(paste0(output_prefix, ".png"), final_plot, width = 14.2, height = 8.1, dpi = 320, bg = "white")
ggsave(paste0(output_prefix, ".svg"), final_plot, width = 14.2, height = 8.1, bg = "white")
