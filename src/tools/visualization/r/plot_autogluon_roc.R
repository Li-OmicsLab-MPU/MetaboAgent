args <- commandArgs(trailingOnly = TRUE)

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

if (length(args) < 7) {
  stop("Usage: Rscript plot_autogluon_roc.R <mode> <roc_csv> <summary_csv> <output_stem> <title> <subtitle> <message> [theme_json]")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}
source(file.path(get_script_dir(), "font_family_helper.R"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

mode <- args[[1]]
roc_csv <- args[[2]]
summary_csv <- args[[3]]
output_stem <- args[[4]]
plot_title <- args[[5]]
plot_subtitle <- args[[6]]
plot_message <- args[[7]]
theme_json <- ifelse(length(args) >= 8, args[[8]], "")
palette_values <- c("#8C1D18", "#1F5AA6", "#2E8B57", "#D18B00", "#6B46C1")
reference_color <- "#94A3B8"
text_color <- "#374151"
grid_color <- "#E5E7EB"
font_family <- resolve_font_family("sans")

if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  palette_values <- theme$palette
  reference_color <- theme$semantic_colors$reference
  text_color <- theme$axis_text_color
  grid_color <- theme$grid_color
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- resolve_font_family(theme$font_family)
  }
}

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

if (mode == "placeholder") {
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.58, label = plot_title, size = 6.0, fontface = "bold") +
    annotate("text", x = 0.5, y = 0.42, label = plot_message, size = 4.2, color = "#475569") +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_void()

  save_plot(p, output_stem, width = 9.6, height = 7.2)
  quit(save = "no")
}

roc_df <- read.csv(roc_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_df <- read.csv(summary_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_df <- summary_df[order(summary_df$rank), , drop = FALSE]

legend_levels <- summary_df$legend_label
roc_df$legend_label <- factor(roc_df$legend_label, levels = legend_levels)
roc_df$rank <- as.integer(roc_df$rank)
roc_df$is_best <- as.logical(roc_df$is_best)

line_values <- setNames(
  c(1.35, 1.05, 1.05, 1.05, 1.05)[seq_along(legend_levels)],
  legend_levels
)

palette_values <- rep(palette_values, length.out = length(legend_levels))
color_values <- setNames(palette_values[seq_along(legend_levels)], legend_levels)

p <- ggplot(roc_df, aes(x = fpr, y = tpr, color = legend_label, group = legend_label)) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "22",
    linewidth = 0.75,
    color = reference_color
  ) +
  geom_line(aes(linewidth = legend_label), lineend = "round") +
  scale_color_manual(values = color_values, breaks = legend_levels, name = NULL) +
  scale_linewidth_manual(values = line_values, breaks = legend_levels, guide = "none") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.2),
    labels = sprintf("%.1f", seq(0, 1, by = 0.2))
  ) +
  scale_y_continuous(
    breaks = seq(0, 1, by = 0.2),
    labels = sprintf("%.1f", seq(0, 1, by = 0.2))
  ) +
  labs(
    title = plot_title,
    subtitle = plot_subtitle,
    x = "False positive rate (1 - specificity)",
    y = "True positive rate (sensitivity)"
  ) +
  guides(
    color = guide_legend(
      order = 1,
      override.aes = list(linewidth = unname(line_values), alpha = 1),
      byrow = TRUE
    )
  ) +
  theme_bw(base_family = font_family, base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "#475569"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = text_color),
    panel.grid.major = element_line(color = grid_color, linewidth = 0.35),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D1D5DB", linewidth = 0.8),
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing.y = unit(3.5, "mm"),
    legend.key.width = unit(10, "mm"),
    legend.key.height = unit(5.2, "mm"),
    legend.text = element_text(size = 9.2, color = text_color),
    legend.background = element_rect(fill = "white", color = NA),
    legend.key = element_rect(fill = "white", color = NA),
    plot.margin = margin(12, 16, 12, 12)
  )

save_plot(p, output_stem, width = 9.8, height = 7.2)
