args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript plot_phase2_optimization_landscape.R <candidates_csv> <layers_csv> <special_csv> <output_stem>")
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

candidates_csv <- args[[1]]
layers_csv <- args[[2]]
special_csv <- args[[3]]
output_stem <- args[[4]]

cand <- read.csv(candidates_csv, check.names = FALSE, stringsAsFactors = FALSE)
layers <- read.csv(layers_csv, check.names = FALSE, stringsAsFactors = FALSE)
special <- read.csv(special_csv, check.names = FALSE, stringsAsFactors = FALSE)

cand$in_pareto_front <- tolower(as.character(cand$in_pareto_front)) == "true"
cand$in_epsilon_feasible_front <- tolower(as.character(cand$in_epsilon_feasible_front)) == "true"
cand$selected_beam <- tolower(as.character(cand$selected_beam)) == "true"

cand$depth_label <- factor(paste0("D", cand$depth), levels = paste0("D", sort(unique(cand$depth))))
layers$depth_label <- factor(paste0("D", layers$depth), levels = paste0("D", sort(unique(layers$depth))))

depth_palette <- c("#1D4E89", "#2A7F8E", "#4BAA7B", "#9BBF30", "#E67E22", "#8C3B2A")
names(depth_palette) <- levels(cand$depth_label)

pareto_df <- cand[cand$in_pareto_front, , drop = FALSE]
beam_df <- cand[cand$selected_beam, , drop = FALSE]
beam_df <- beam_df[order(beam_df$depth, beam_df$perf), , drop = FALSE]
beam_df <- beam_df[!duplicated(beam_df$depth), , drop = FALSE]
winner_df <- subset(special, point_type == "Phase 2 winner")
pareto_boundary_df <- pareto_df[order(pareto_df$cost, pareto_df$perf), , drop = FALSE]

x_focus_max <- max(cand$cost, na.rm = TRUE) * 1.10
y_focus_min <- min(cand$perf, na.rm = TRUE) - 0.01
y_focus_max <- min(0.985, max(cand$perf, na.rm = TRUE) + 0.01)

winner_x <- winner_df$cost[1]
winner_y <- winner_df$perf[1]
zoom_xmin <- max(0, winner_x - 0.018)
zoom_xmax <- min(x_focus_max, winner_x + 0.020)
zoom_ymin <- max(y_focus_min, winner_y - 0.012)
zoom_ymax <- min(y_focus_max, winner_y + 0.010)

main_plot <- ggplot(cand, aes(x = cost, y = perf)) +
  stat_density_2d(
    aes(fill = after_stat(level)),
    geom = "polygon",
    alpha = 0.12,
    contour = TRUE,
    bins = 5,
    show.legend = FALSE
  ) +
  scale_fill_gradient(low = "#FBFCFE", high = "#D97986", guide = "none") +
  geom_point(
    aes(size = independence, color = bio),
    alpha = 0.62,
    stroke = 0,
    shape = 21,
    show.legend = TRUE
  ) +
  geom_step(
    data = pareto_boundary_df,
    aes(x = cost, y = perf, linetype = "Pareto feasible boundary"),
    inherit.aes = FALSE,
    linewidth = 0.95,
    color = "#B85C6B",
    direction = "vh",
    alpha = 0.85
  ) +
  geom_path(
    data = beam_df,
    aes(x = cost, y = perf, linetype = "Selected beam trajectory"),
    inherit.aes = FALSE,
    linewidth = 1.35,
    color = "#2E8B57",
    alpha = 0.95,
    arrow = arrow(type = "closed", length = unit(0.10, "inches"))
  ) +
  geom_point(
    data = pareto_df,
    aes(x = cost, y = perf),
    inherit.aes = FALSE,
    shape = 23,
    stroke = 0.85,
    size = 2.9,
    fill = "#F7E6A7",
    color = "#8C5A2B"
  ) +
  geom_text(
    data = beam_df,
    aes(x = cost, y = perf, label = paste0("D", depth)),
    inherit.aes = FALSE,
    nudge_x = 0.0026,
    nudge_y = 0.0038,
    size = 3.1,
    color = "#6E2C00",
    fontface = "bold"
  ) +
  geom_point(
    data = winner_df,
    aes(x = cost, y = perf),
    inherit.aes = FALSE,
    shape = 8,
    size = 9.4,
    stroke = 2.2,
    color = "white"
  ) +
  geom_point(
    data = winner_df,
    aes(x = cost, y = perf),
    inherit.aes = FALSE,
    shape = 8,
    size = 6.6,
    stroke = 1.4,
    color = "#D4A017"
  ) +
  scale_color_gradient(
    low = "#DDF3E4",
    high = "#0B6E4F",
    name = "Biological support"
  ) +
  scale_size_continuous(
    range = c(1.2, 4.2),
    name = "1 - f_corr"
  ) +
  scale_linetype_manual(
    values = c(
      "Selected beam trajectory" = "solid",
      "Pareto feasible boundary" = "22"
    ),
    name = "Path"
  ) +
  labs(
    title = "Phase 2 Optimization Landscape",
    subtitle = "Search space projected onto predictive performance and clinical cost",
    x = "Clinical cost (f_cost)",
    y = "Predictive performance (AUC)"
  ) +
  coord_cartesian(xlim = c(0, x_focus_max), ylim = c(y_focus_min, y_focus_max), expand = FALSE) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 10.5, hjust = 0.5, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#E5E7EB", linewidth = 0.35),
    legend.position = "right",
    legend.box = "vertical",
    axis.text = element_text(color = "#334155"),
    plot.margin = margin(8, 8, 4, 4)
  ) +
  guides(
    color = guide_colorbar(order = 1),
    size = guide_legend(order = 2),
    linetype = guide_legend(order = 3, override.aes = list(color = c("#216E5B", "#B85C6B"), linewidth = c(1.2, 0.95)))
  )

winner_inset <- ggplot(cand, aes(x = cost, y = perf)) +
  geom_point(
    aes(size = independence, color = bio),
    alpha = 0.75,
    stroke = 0,
    shape = 16
  ) +
  geom_path(
    data = beam_df,
    aes(x = cost, y = perf),
    inherit.aes = FALSE,
    linewidth = 0.9,
    color = "#2E8B57",
    alpha = 0.95
  ) +
  geom_point(
    data = winner_df,
    aes(x = cost, y = perf),
    inherit.aes = FALSE,
    shape = 8,
    size = 6.2,
    stroke = 1.8,
    color = "white"
  ) +
  geom_point(
    data = winner_df,
    aes(x = cost, y = perf),
    inherit.aes = FALSE,
    shape = 8,
    size = 4.5,
    stroke = 1.2,
    color = "#D4A017"
  ) +
  scale_color_gradient(low = "#DDF3E4", high = "#0B6E4F", guide = "none") +
  scale_size_continuous(range = c(1.2, 3.2), guide = "none") +
  coord_cartesian(
    xlim = c(zoom_xmin, zoom_xmax),
    ylim = c(zoom_ymin, zoom_ymax),
    expand = FALSE
  ) +
  labs(title = "Winner region ×3") +
  theme_bw(base_family = "sans", base_size = 8.5) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 8.8, color = "#334155"),
    axis.title = element_text(size = 7.0, face = "bold", color = "#475569"),
    axis.text = element_text(size = 7.2, color = "#475569"),
    panel.grid = element_line(color = "#E5E7EB", linewidth = 0.25),
    plot.margin = margin(3, 3, 3, 3)
  ) +
  xlab("f_cost") +
  ylab("AUC")

main_plot_composed <- ggdraw() +
  draw_plot(main_plot, x = 0, y = 0, width = 1, height = 1) +
  draw_plot(winner_inset, x = 0.66, y = 0.13, width = 0.20, height = 0.21)

top_density <- ggplot(cand, aes(x = cost, group = depth_label, fill = depth_label, color = depth_label)) +
  geom_density(alpha = 0.28, linewidth = 0.62, adjust = 1.05) +
  scale_fill_manual(values = depth_palette, guide = "none") +
  scale_color_manual(values = depth_palette, guide = "none") +
  labs(x = NULL, y = NULL) +
  ggtitle("Marginal cost distribution") +
  theme_bw(base_family = "sans", base_size = 9) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 9.5, face = "bold", color = "#475569"),
    legend.position = "none",
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    strip.background = element_blank(),
    plot.margin = margin(0, 10, 2, 42)
  )

right_density <- ggplot(cand, aes(x = factor("AUC"), y = perf, fill = depth_label, color = depth_label)) +
  geom_violin(
    linewidth = 0.35,
    alpha = 0.82,
    width = 0.86
  ) +
  geom_boxplot(
    width = 0.12,
    outlier.shape = NA,
    fill = "white",
    color = "#4B5563",
    linewidth = 0.30
  ) +
  geom_point(
    position = position_jitter(width = 0.07, height = 0),
    size = 0.95,
    alpha = 0.85,
    stroke = 0
  ) +
  facet_grid(depth_label ~ ., scales = "free_y", switch = "y") +
  coord_flip() +
  scale_fill_manual(values = depth_palette, guide = "none") +
  scale_color_manual(values = depth_palette, guide = "none") +
  labs(x = NULL, y = NULL, title = "AUC distribution by depth") +
  theme_bw(base_family = "sans", base_size = 9) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 9.5, face = "bold", color = "#475569"),
    legend.position = "none",
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "#F8FAFC", color = "#CBD5E1", linewidth = 0.35),
    strip.text.y.left = element_text(angle = 0, face = "bold", color = "#334155", size = 8.2),
    strip.placement = "outside",
    panel.spacing.y = unit(0.12, "lines"),
    plot.margin = margin(28, 0, 22, 0)
  )

final_plot <- plot_grid(
  plot_grid(NULL, top_density, ncol = 2, rel_widths = c(0.08, 1)),
  plot_grid(
    main_plot_composed,
    right_density,
    ncol = 2,
    rel_widths = c(1, 0.22)
  ),
  ncol = 1,
  rel_heights = c(0.18, 1)
)

save_plot <- function(plot_obj, output_stem, width, height) {
  ggsave(paste0(output_stem, ".pdf"), plot_obj, width = width, height = height, device = cairo_pdf, dpi = 300)
  ggsave(paste0(output_stem, ".png"), plot_obj, width = width, height = height, dpi = 300, bg = "white")
  svg(paste0(output_stem, ".svg"), width = width, height = height, bg = "white")
  print(plot_obj)
  dev.off()
}

save_plot(final_plot, output_stem, width = 11.8, height = 8.3)
