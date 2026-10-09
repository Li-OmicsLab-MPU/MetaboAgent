args <- commandArgs(trailingOnly = TRUE)

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

if (length(args) < 3) {
  stop("Usage: Rscript plot_phase1_method_support_dot_matrix.R <long_csv> <output_stem> <title> [theme_json]")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(grid)
})

long_csv <- args[[1]]
output_stem <- args[[2]]
plot_title <- args[[3]]
theme_json <- ifelse(length(args) >= 4, args[[4]], "")
subtitle_text <- ifelse(length(args) >= 5, args[[5]], "")
font_family <- "sans"

text_color <- "#1F2937"
axis_text_color <- "#374151"
grid_color <- "#E5E7EB"
minor_grid_color <- "#F3F4F6"
panel_border_color <- "#D1D5DB"
reference_color <- "#BBBDC0"
frequency_colors <- c("#F5F7FB", "#D7E8F5", "#9DCBE6", "#4F97C8", "#202754")
if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  text_color <- theme$text_color
  axis_text_color <- theme$axis_text_color
  grid_color <- theme$grid_color
  minor_grid_color <- theme$grid_color
  panel_border_color <- theme$panel_border_color
  reference_color <- theme$semantic_colors$reference
  frequency_colors <- grDevices::colorRampPalette(c("#F7FAFC", theme$semantic_colors$phase1, theme$semantic_colors$highlight))(5)
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- theme$font_family
  }
}

wrap_feature_label <- function(x, width = 26) {
  if (!nzchar(x)) return(x)
  x <- gsub(" / ", " /\n", x, fixed = TRUE)
  paste(strwrap(x, width = width), collapse = "\n")
}

long_df <- read.csv(long_csv, check.names = FALSE, stringsAsFactors = FALSE)
long_df$display_label <- vapply(long_df$display_label, wrap_feature_label, character(1))

feature_order <- unique(long_df[, c("display_label", "feature_order")])
feature_levels <- rev(feature_order$display_label[order(feature_order$feature_order)])
long_df$display_label <- factor(long_df$display_label, levels = feature_levels)
long_df$method_label <- factor(
  long_df$method_label,
  levels = c("Elastic Net", "Lasso", "Random Forest", "LightGBM", "mRMR", "Welch t-test", "FDR/effect-size")
)

p <- ggplot(long_df, aes(x = method_label, y = display_label)) +
  geom_point(
    aes(size = method_frequency, color = method_frequency),
    shape = 16,
    alpha = 0.98
  ) +
  scale_size_continuous(
    range = c(2.8, 10.8),
    limits = c(0, 1),
    breaks = c(0.25, 0.50, 0.75, 1.00),
    labels = label_percent(accuracy = 1),
    name = "Per-method\nfrequency"
  ) +
  scale_color_gradientn(
    colours = frequency_colors,
    limits = c(0, 1),
    breaks = c(0.25, 0.50, 0.75, 1.00),
    labels = label_percent(accuracy = 1),
    name = "Per-method\nfrequency"
  ) +
  guides(
    color = guide_colourbar(
      order = 1,
      barheight = unit(45, "mm"),
      barwidth = unit(5.5, "mm"),
      frame.colour = reference_color,
      ticks.colour = axis_text_color
    ),
    size = guide_legend(
      order = 2,
      override.aes = list(color = frequency_colors[length(frequency_colors)], alpha = 1)
    )
  ) +
  labs(
    title = plot_title,
    subtitle = subtitle_text,
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_family = font_family, base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = axis_text_color),
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, color = axis_text_color, size = 9.5),
    axis.text.y = element_text(color = text_color, size = 9.5),
    panel.grid.major.x = element_line(color = grid_color, linewidth = 0.35),
    panel.grid.major.y = element_line(color = minor_grid_color, linewidth = 0.30),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = panel_border_color, linewidth = 0.8),
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing.y = unit(4, "mm"),
    legend.background = element_rect(fill = "white", color = NA),
    legend.key = element_rect(fill = "white", color = NA),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 8.5),
    plot.margin = margin(12, 18, 12, 14)
  )

save_plot <- function(plot_obj, output_stem, width, height) {
  cairo_pdf(paste0(output_stem, ".pdf"), width = width, height = height, family = font_family)
  print(plot_obj)
  dev.off()

  png(paste0(output_stem, ".png"), width = width * 300, height = height * 300, res = 300, type = "cairo")
  print(plot_obj)
  dev.off()

  svg(paste0(output_stem, ".svg"), width = width, height = height)
  print(plot_obj)
  dev.off()
}

save_plot(p, output_stem, width = 10.6, height = 7.8)
