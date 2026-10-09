args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Usage: Rscript plot_phase2_pairwise_tradeoff_matrix.R <candidates_csv> <special_csv> <output_stem>")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(GGally)
  library(ggplot2)
})

candidates_csv <- args[[1]]
special_csv <- args[[2]]
output_stem <- args[[3]]

cand <- read.csv(candidates_csv, check.names = FALSE, stringsAsFactors = FALSE)
special <- read.csv(special_csv, check.names = FALSE, stringsAsFactors = FALSE)

plot_df <- data.frame(
  AUC = cand$perf,
  Biology = cand$bio,
  Independence = 1 - cand$corr,
  ClinicalBenefit = 1 - cand$cost,
  Depth = factor(cand$depth)
)

depth_palette <- c("#4C78A8", "#5F8BC5", "#76A5AF", "#F2B134", "#E67E22", "#C0392B")
names(depth_palette) <- levels(plot_df$Depth)

upper_fn <- function(data, mapping, ...) {
  x <- GGally::eval_data_col(data, mapping$x)
  y <- GGally::eval_data_col(data, mapping$y)
  ct <- suppressWarnings(cor.test(x, y, method = "spearman"))
  label <- sprintf("rho = %.2f\np = %.3g", unname(ct$estimate), ct$p.value)
  ggplot(data = data, mapping = mapping) +
    annotate("text", x = mean(range(x, na.rm = TRUE)), y = mean(range(y, na.rm = TRUE)), label = label, size = 3.3, color = "#334155") +
    theme_void()
}

lower_fn <- function(data, mapping, ...) {
  ggplot(data = data, mapping = mapping) +
    stat_density_2d(aes(fill = after_stat(level)), geom = "polygon", alpha = 0.18, contour = TRUE, bins = 6, show.legend = FALSE) +
    scale_fill_gradient(low = "#F8FAFC", high = "#D1D5DB") +
    geom_point(aes(color = Depth), alpha = 0.55, size = 1.5) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 0.7, color = "#111827") +
    theme_bw(base_family = "sans", base_size = 9.5) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#E5E7EB", linewidth = 0.25)
    )
}

diag_fn <- function(data, mapping, ...) {
  ggplot(data = data, mapping = mapping) +
    geom_histogram(fill = "#CBD5E1", color = "white", bins = 18) +
    geom_density(color = "#0F172A", linewidth = 0.7, alpha = 0.8) +
    theme_bw(base_family = "sans", base_size = 9.5) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#E5E7EB", linewidth = 0.25)
    )
}

g <- ggpairs(
  plot_df,
  columns = 1:4,
  upper = list(continuous = wrap(upper_fn)),
  lower = list(continuous = wrap(lower_fn)),
  diag = list(continuous = wrap(diag_fn)),
  mapping = aes(color = Depth)
) +
  theme_bw(base_family = "sans", base_size = 10.5) +
  theme(
    strip.text = element_text(face = "bold", size = 10.5),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

for (i in 1:4) {
  for (j in 1:4) {
    p <- getPlot(g, i, j)
    if (!is.null(p) && inherits(p, "ggplot") && i > j) {
      p <- p + scale_color_manual(values = depth_palette, drop = FALSE)
      winner <- subset(special, point_type == "Phase 2 winner")
      if (nrow(winner) == 1) {
        winner_map <- c(winner$perf, winner$bio, 1 - winner$corr, 1 - winner$cost)
        p <- p +
          geom_point(
            inherit.aes = FALSE,
            data = data.frame(
              x = winner_map[j],
              y = winner_map[i]
            ),
            aes(x = x, y = y),
            shape = 8,
            size = 3.2,
            color = "#D4A017"
          )
      }
      g <- putPlot(g, p, i, j)
    }
  }
}

title_grob <- ggplot() +
  annotate("text", x = 0.5, y = 0.62, label = "Four-Objective Pairwise Tradeoff Structure", size = 6.0, fontface = "bold") +
  annotate("text", x = 0.5, y = 0.28, label = "All explored candidate panels, search-depth encoding, density structure and pairwise objective coupling", size = 3.4, color = "#475569") +
  theme_void()

final_plot <- cowplot::plot_grid(title_grob, g, ncol = 1, rel_heights = c(0.08, 1))

ggsave(paste0(output_stem, ".pdf"), final_plot, width = 11.8, height = 10.6, device = cairo_pdf, dpi = 300)
ggsave(paste0(output_stem, ".png"), final_plot, width = 11.8, height = 10.6, dpi = 300, bg = "white")
svg(paste0(output_stem, ".svg"), width = 11.8, height = 10.6, bg = "white")
print(final_plot)
dev.off()
