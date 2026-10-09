args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript plot_phase2_pathway_annotation_preview.R <annotation_csv> <contrib_csv> <output_prefix>")
}

suppressPackageStartupMessages({
  lib_path <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
  if (nzchar(lib_path)) {
    .libPaths(unique(c(lib_path, .libPaths())))
  }
  library(ggplot2)
  library(dplyr)
  library(scales)
})

annotation_csv <- args[[1]]
contrib_csv <- args[[2]]
output_prefix <- args[[3]]

anno_df <- read.csv(annotation_csv, stringsAsFactors = FALSE, check.names = FALSE)
contrib_df <- read.csv(contrib_csv, stringsAsFactors = FALSE, check.names = FALSE)

if (nrow(anno_df) == 0 || nrow(contrib_df) == 0) {
  stop("Annotation or contribution table is empty.")
}

plot_df <- anno_df %>%
  left_join(
    contrib_df %>%
      transmute(
        feature,
        delta_keep_perf = as.numeric(delta_keep_perf),
        importance_abs = abs(as.numeric(delta_keep_perf)),
        support_score = as.numeric(winner_support_score)
      ),
    by = "feature"
  ) %>%
  mutate(
    importance_abs = ifelse(is.na(importance_abs), 0.01, importance_abs),
    support_score = ifelse(is.na(support_score), 0, support_score),
    feature_label = display_name,
    y = seq_len(n())
  )

plot_df <- plot_df %>% arrange(importance_abs)

wrap_text <- function(x, width = 18) {
  vapply(
    as.character(x),
    function(item) paste(strwrap(item, width = width), collapse = "\n"),
    character(1)
  )
}

track_palette <- c(
  "Hydroxy fatty acid" = "#3B82F6",
  "Epoxy fatty acid" = "#F59E0B",
  "Dihydroxy fatty acid" = "#10B981",
  "Prostaglandin" = "#BE123C",
  "EPA (omega-3)" = "#4C78A8",
  "DGLA / eicosatrienoate" = "#F58518",
  "Linoleate-derived oxylipin" = "#54A24B",
  "Arachidonate (omega-6)" = "#E45756",
  "Lipoxygenase-derived oxylipin" = "#60A5FA",
  "CYP epoxygenase / epoxide axis" = "#F59E0B",
  "sEH / diol oxylipin axis" = "#34D399",
  "Cyclooxygenase prostaglandin axis" = "#FB7185",
  "Omega-3 oxylipin signaling" = "#2563EB",
  "Epoxygenase-eicosanoid signaling" = "#D97706",
  "Oxylipin remodeling and diol metabolism" = "#059669",
  "Prostaglandin signaling" = "#BE123C",
  "PUFA-derived lipid mediators" = "#4C78A8",
  "Eicosanoid / epoxy-fatty-acid metabolism" = "#F58518",
  "Oxylipin metabolism" = "#54A24B",
  "Cyclooxygenase-derived prostanoids" = "#E45756"
)

annotation_long <- bind_rows(
  plot_df %>% transmute(display_name, y, track = "Primary class", value = primary_class),
  plot_df %>% transmute(display_name, y, track = "Precursor pool", value = upstream_fatty_acid),
  plot_df %>% transmute(display_name, y, track = "Enzyme axis", value = enzyme_axis),
  plot_df %>% transmute(display_name, y, track = "Pathway module", value = pathway_module),
  plot_df %>% transmute(display_name, y, track = "Pathway group", value = pathway_group)
) %>%
  mutate(
    track = factor(track, levels = c("Primary class", "Precursor pool", "Enzyme axis", "Pathway module", "Pathway group")),
    x = c(1, 2, 3, 4, 5)[as.integer(track)],
    value_wrapped = case_when(
      track == "Primary class" ~ wrap_text(value, 16),
      track == "Precursor pool" ~ wrap_text(value, 18),
      track == "Enzyme axis" ~ wrap_text(value, 18),
      track == "Pathway module" ~ wrap_text(value, 22),
      TRUE ~ wrap_text(value, 20)
    )
  )

for (nm in unique(annotation_long$value)) {
  if (!(nm %in% names(track_palette))) {
    track_palette[[nm]] <- "#94A3B8"
  }
}

track_headers <- data.frame(
  x = c(-1.25, 0, 1, 2, 3, 4, 5),
  y = max(plot_df$y) + 0.72,
  label = c("Relative\nAUC gain", "Winner metabolite", "Primary class", "Precursor pool", "Enzyme axis", "Pathway module", "Pathway group")
)

segment_df <- data.frame(
  x = c(0.55, 1.55, 2.55, 3.55),
  xend = c(0.85, 1.85, 2.85, 3.85)
)

main_plot <- ggplot() +
  geom_segment(
    data = plot_df,
    aes(x = -importance_abs, xend = 0, y = y, yend = y),
    linewidth = 1.2,
    color = "#94A3B8",
    lineend = "round"
  ) +
  geom_point(
    data = plot_df,
    aes(x = -importance_abs, y = y, size = importance_abs),
    fill = "#A7493B",
    color = "white",
    shape = 21,
    stroke = 0.55
  ) +
  geom_label(
    data = plot_df,
    aes(x = -importance_abs - 0.008, y = y, label = sprintf("%.1f%%", 100 * importance_abs / sum(importance_abs))),
    size = 3.2,
    hjust = 1,
    linewidth = 0,
    fill = alpha("white", 0.84),
    color = "#334155",
    fontface = "bold",
    label.padding = unit(0.1, "lines")
  ) +
  geom_text(
    data = plot_df,
    aes(x = 0, y = y, label = feature_label),
    hjust = 0,
    nudge_x = 0.07,
    size = 4.0,
    family = "sans",
    fontface = "bold",
    color = "#111827"
  ) +
  geom_segment(
    data = segment_df,
    aes(x = x, xend = xend, y = plot_df$y[1], yend = plot_df$y[1]),
    inherit.aes = FALSE,
    color = alpha("#CBD5E1", 0)
  ) +
  geom_curve(
    data = plot_df,
    aes(x = 0.38, xend = 0.86, y = y, yend = y),
    curvature = 0.08,
    color = alpha("#CBD5E1", 0.95),
    linewidth = 0.45
  ) +
  geom_curve(
    data = plot_df,
    aes(x = 1.14, xend = 1.86, y = y, yend = y),
    curvature = 0.08,
    color = alpha("#CBD5E1", 0.95),
    linewidth = 0.45
  ) +
  geom_curve(
    data = plot_df,
    aes(x = 2.14, xend = 2.86, y = y, yend = y),
    curvature = 0.08,
    color = alpha("#CBD5E1", 0.95),
    linewidth = 0.45
  ) +
  geom_curve(
    data = plot_df,
    aes(x = 3.14, xend = 3.86, y = y, yend = y),
    curvature = 0.08,
    color = alpha("#CBD5E1", 0.95),
    linewidth = 0.45
  ) +
  geom_curve(
    data = plot_df,
    aes(x = 4.14, xend = 4.86, y = y, yend = y),
    curvature = 0.08,
    color = alpha("#CBD5E1", 0.95),
    linewidth = 0.45
  ) +
  geom_tile(
    data = annotation_long,
    aes(x = x, y = y, fill = value),
    width = 0.88,
    height = 0.72,
    color = "white",
    linewidth = 0.45
  ) +
  geom_text(
    data = annotation_long,
    aes(x = x, y = y, label = value_wrapped),
    size = 2.75,
    family = "sans",
    lineheight = 0.92,
    color = "#111827"
  ) +
  geom_text(
    data = track_headers,
    aes(x = x, y = y, label = label),
    size = 3.35,
    family = "sans",
    fontface = "bold",
    color = "#374151",
    lineheight = 0.94
  ) +
  annotate(
    "text",
    x = 2.5,
    y = max(plot_df$y) + 1.45,
    label = "Phase 2 Winner Panel Pathway Annotation Map",
    family = "sans",
    fontface = "bold",
    size = 5.2,
    color = "#111827"
  ) +
  annotate(
    "text",
    x = 2.5,
    y = max(plot_df$y) + 1.12,
    label = "Curated annotation of oxylipin/eicosanoid biology; left track scaled by phase2 feature contribution",
    family = "sans",
    size = 3.5,
    color = "#6B7280"
  ) +
  scale_fill_manual(values = track_palette, guide = "none") +
  scale_size_continuous(range = c(4.2, 8.8), guide = "none") +
  coord_cartesian(
    xlim = c(-0.17, 5.52),
    ylim = c(0.4, max(plot_df$y) + 1.7),
    clip = "off"
  ) +
  theme_void(base_family = "sans") +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(18, 16, 14, 24)
  )

ggsave(paste0(output_prefix, ".pdf"), main_plot, width = 13.6, height = 7.6, device = cairo_pdf)
ggsave(paste0(output_prefix, ".png"), main_plot, width = 13.6, height = 7.6, dpi = 320)
if (requireNamespace("svglite", quietly = TRUE)) {
  ggsave(paste0(output_prefix, ".svg"), main_plot, width = 13.6, height = 7.6)
}
