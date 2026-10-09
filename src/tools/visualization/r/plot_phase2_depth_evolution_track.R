args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript plot_phase2_depth_evolution_track.R <candidates_csv> <layers_csv> <special_csv> <output_stem>")
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

depth_levels <- as.character(sort(unique(cand$depth)))
cand$depth_label <- factor(paste0("D", cand$depth), levels = rev(paste0("D", depth_levels)))
layers$depth_label <- factor(paste0("D", layers$depth), levels = rev(paste0("D", depth_levels)))

depth_palette <- c("#4C78A8", "#5F8BC5", "#76A5AF", "#F2B134", "#E67E22", "#C0392B")
names(depth_palette) <- paste0("D", sort(unique(cand$depth)))

pareto_best <- do.call(
  rbind,
  lapply(split(cand, cand$depth), function(df) {
    df <- df[df$in_pareto_front, , drop = FALSE]
    if (nrow(df) == 0) {
      return(NULL)
    }
    df[which.max(df$perf), c("depth", "perf", "bio", "cost", "corr", "n_features")]
  })
)
pareto_best <- as.data.frame(pareto_best)
pareto_best$depth <- as.integer(as.character(pareto_best$depth))
pareto_best$depth_label <- factor(paste0("D", pareto_best$depth), levels = levels(cand$depth_label))
pareto_best$neglog10_p <- -log10(layers$delong_p_value)

track_plot <- ggplot(cand, aes(x = perf, y = depth_label, fill = depth_label)) +
  geom_violin(scale = "width", alpha = 0.75, color = NA, width = 0.88) +
  geom_rug(
    data = cand[cand$in_pareto_front, , drop = FALSE],
    aes(x = perf, y = depth_label),
    inherit.aes = FALSE,
    sides = "b",
    alpha = 0.55,
    color = "#6B1E1E"
  ) +
  geom_point(
    data = cand[cand$in_pareto_front, , drop = FALSE],
    aes(x = perf, y = depth_label),
    inherit.aes = FALSE,
    shape = 124,
    size = 3.8,
    color = "#8E1B1B",
    alpha = 0.9
  ) +
  geom_point(
    data = pareto_best,
    aes(x = perf, y = depth_label),
    inherit.aes = FALSE,
    shape = 23,
    size = 3.7,
    fill = "#FFF3B0",
    color = "#7C2D12",
    stroke = 0.7
  ) +
  geom_segment(
    data = pareto_best[-nrow(pareto_best), ],
    aes(
      x = perf,
      xend = pareto_best$perf[-1],
      y = depth_label,
      yend = pareto_best$depth_label[-1]
    ),
    inherit.aes = FALSE,
    color = "#2F855A",
    linewidth = 0.9,
    arrow = arrow(length = unit(0.12, "inches"), type = "closed")
  ) +
  geom_text(
    data = layers,
    aes(x = max(cand$perf) + 0.005, y = depth_label,
        label = paste0("N=", expanded_count, " | Pareto=", pareto_count)),
    inherit.aes = FALSE,
    hjust = 0,
    size = 3.0,
    color = "#475569"
  ) +
  scale_fill_manual(values = depth_palette, name = "Depth") +
  labs(
    title = "Depth Evolution Track",
    subtitle = "Sequential Pareto-guided search trajectory across depths",
    x = "Predictive performance (AUC)",
    y = NULL
  ) +
  theme_bw(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 10.5, hjust = 0.5, color = "#475569"),
    axis.text = element_text(color = "#334155"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = 0.35),
    legend.position = "none",
    plot.margin = margin(4, 52, 4, 4)
  ) +
  coord_cartesian(xlim = c(min(cand$perf) - 0.01, max(cand$perf) + 0.045))

pval_df <- layers
pval_df$depth_label <- factor(paste0("D", pval_df$depth), levels = levels(cand$depth_label))
pval_df$neglog10_p <- -log10(pval_df$delong_p_value)

pval_plot <- ggplot(pval_df, aes(x = depth_label, y = neglog10_p, group = 1)) +
  geom_hline(yintercept = -log10(0.05), linetype = "22", linewidth = 0.7, color = "#B91C1C") +
  geom_line(color = "#9F1239", linewidth = 0.9) +
  geom_point(size = 2.4, color = "#9F1239") +
  labs(y = expression(-log[10](p)), x = NULL) +
  theme_bw(base_family = "sans", base_size = 9.5) +
  theme(
    axis.text = element_text(color = "#334155"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#E5E7EB", linewidth = 0.3),
    plot.margin = margin(4, 4, 2, 40)
  ) +
  coord_flip()

final_plot <- plot_grid(
  pval_plot,
  track_plot,
  ncol = 1,
  rel_heights = c(0.34, 1)
)

save_plot <- function(plot_obj, output_stem, width, height) {
  ggsave(paste0(output_stem, ".pdf"), plot_obj, width = width, height = height, device = cairo_pdf, dpi = 300)
  ggsave(paste0(output_stem, ".png"), plot_obj, width = width, height = height, dpi = 300, bg = "white")
  svg(paste0(output_stem, ".svg"), width = width, height = height, bg = "white")
  print(plot_obj)
  dev.off()
}

save_plot(final_plot, output_stem, width = 11.4, height = 8.6)
