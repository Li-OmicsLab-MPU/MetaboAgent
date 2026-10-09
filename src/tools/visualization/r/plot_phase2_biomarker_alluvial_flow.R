args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript plot_phase2_biomarker_alluvial_flow.R <summary_csv> <output_prefix>")
}

suppressPackageStartupMessages({
  lib_path <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
  if (nzchar(lib_path)) {
    .libPaths(unique(c(lib_path, .libPaths())))
  }
  library(ggplot2)
  library(ggalluvial)
  library(cowplot)
  library(dplyr)
})

summary_csv <- args[[1]]
output_prefix <- args[[2]]

df <- read.csv(summary_csv, stringsAsFactors = FALSE, check.names = FALSE)
if (nrow(df) == 0) {
  stop("No rows found in Figure C summary data.")
}

df$prior_group <- factor(df$prior_group, levels = c("Prior-supported", "Data-driven"))
df$phase2_fate <- factor(df$phase2_fate, levels = c("Winner", "Pareto-supporting", "Search-only"))
df$feature_label <- factor(df$feature_label, levels = rev(unique(df$feature_label)))

category_levels <- unique(df$feature_category)
category_palette <- c(
  "Amino acid-like" = "#2F6B8A",
  "Purine / Nucleoside" = "#A74F3B",
  "Carbohydrate-like" = "#C3A035",
  "Sulfur / Lactone" = "#6B8E6E",
  "Other / Named" = "#7A7A7A"
)
fallback_cols <- c("#7C90B7", "#4D8B74", "#C9795E", "#C5B358", "#7B6D8D", "#5F9EA0")
for (i in seq_along(category_levels)) {
  level <- category_levels[[i]]
  if (is.null(category_palette[[level]])) {
    category_palette[[level]] <- fallback_cols[((i - 1) %% length(fallback_cols)) + 1]
  }
}

axis_labels <- c("Prior support", "Phase 1 panel feature", "Phase 2 fate")

base_plot <- ggplot(
  df,
  aes(
    axis1 = prior_group,
    axis2 = feature_label,
    axis3 = phase2_fate,
    y = flow_weight
  )
) +
  geom_alluvium(
    aes(fill = feature_category),
    width = 0.28,
    alpha = 0.88,
    knot.pos = 0.42,
    color = alpha("#FFFFFF", 0.32),
    decreasing = FALSE
  ) +
  geom_stratum(
    width = 0.28,
    fill = "white",
    color = "#CBD5E1",
    linewidth = 0.45
  ) +
  geom_text(
    stat = "stratum",
    aes(label = after_stat(stratum)),
    size = 3.2,
    family = "sans",
    color = "#1F2937",
    lineheight = 0.92
  ) +
  scale_x_discrete(labels = axis_labels, expand = c(0.10, 0.10)) +
  scale_fill_manual(values = category_palette, name = "Feature category") +
  labs(
    title = "Figure C. Biomarker Alluvial Flow",
    subtitle = "Prior support, feature identity, and final Phase 2 fate for the Phase 1 panel",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_family = "sans", base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14, color = "#111827"),
    plot.subtitle = element_text(hjust = 0.5, size = 10, color = "#4B5563"),
    axis.text.y = element_blank(),
    axis.text.x = element_text(size = 10.5, face = "bold", color = "#374151"),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(size = 9.5, face = "bold"),
    legend.text = element_text(size = 8.8),
    plot.margin = margin(12, 14, 8, 14)
  )

fate_summary <- df %>%
  count(phase2_fate, name = "n") %>%
  mutate(
    phase2_fate = factor(phase2_fate, levels = c("Winner", "Pareto-supporting", "Search-only")),
    label = paste0(as.character(phase2_fate), "\n(n=", n, ")")
  )

summary_plot <- ggplot(fate_summary, aes(x = phase2_fate, y = n, fill = phase2_fate)) +
  geom_col(width = 0.58, color = "white", linewidth = 0.35) +
  geom_text(aes(label = n), vjust = -0.35, size = 3.5, color = "#111827", family = "sans") +
  scale_fill_manual(
    values = c("Winner" = "#A74F3B", "Pareto-supporting" = "#D5A24C", "Search-only" = "#94A3B8"),
    guide = "none"
  ) +
  scale_x_discrete(labels = setNames(fate_summary$label, fate_summary$phase2_fate)) +
  labs(x = NULL, y = "Feature count") +
  theme_minimal(base_family = "sans", base_size = 10) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(face = "bold", color = "#374151"),
    axis.title.y = element_text(size = 9.5, face = "bold"),
    plot.margin = margin(0, 14, 10, 14)
  )

final_plot <- plot_grid(
  base_plot,
  summary_plot,
  ncol = 1,
  rel_heights = c(4.8, 1.4),
  align = "v"
)

ggsave(paste0(output_prefix, ".pdf"), final_plot, width = 11.5, height = 8.6, device = cairo_pdf)
ggsave(paste0(output_prefix, ".png"), final_plot, width = 11.5, height = 8.6, dpi = 320)
ggsave(paste0(output_prefix, ".svg"), final_plot, width = 11.5, height = 8.6)
