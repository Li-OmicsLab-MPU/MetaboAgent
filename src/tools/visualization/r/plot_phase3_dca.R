args <- commandArgs(trailingOnly = TRUE)

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

if (length(args) < 3) {
  stop("Usage: Rscript plot_phase3_dca.R <curve_csv> <summary_csv> <output_stem> [theme_json] [title]")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}
source(file.path(get_script_dir(), "font_family_helper.R"))

suppressPackageStartupMessages({
  library(ggplot2)
})

primary_color <- "#8C1D18"
treat_all_color <- "#6B7280"
treat_none_color <- "#B8C0CC"
text_color <- "#1F2937"
grid_color <- "#E9EDF2"

curve_csv <- args[[1]]
summary_csv <- args[[2]]
output_stem <- args[[3]]
theme_json <- ifelse(length(args) >= 4, args[[4]], "")
title_override <- ifelse(length(args) >= 5, args[[5]], "")
font_family <- "sans"

if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  primary_color <- theme$semantic_colors$winner
  treat_all_color <- theme$semantic_colors$baseline
  treat_none_color <- theme$semantic_colors$reference
  text_color <- theme$text_color
  grid_color <- theme$grid_color
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- theme$font_family
  }
}
font_family <- resolve_font_family(font_family)

save_plot <- function(plot_obj, output_stem, width, height) {
  cairo_pdf(paste0(output_stem, ".pdf"), width = width, height = height, family = font_family, bg = "white")
  print(plot_obj)
  dev.off()

  png(paste0(output_stem, ".png"), width = width * 300, height = height * 300, res = 300, bg = "white", type = "cairo")
  print(plot_obj)
  dev.off()

  svg(paste0(output_stem, ".svg"), width = width, height = height, bg = "white")
  print(plot_obj)
  dev.off()
}

curve_df <- read.csv(curve_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_df <- read.csv(summary_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_row <- summary_df[1, , drop = FALSE]

feature_count <- as.integer(summary_row$feature_count[[1]])
scenario_name <- as.character(summary_row$scenario_name[[1]])
default_threshold <- as.numeric(summary_row$default_threshold[[1]])
companion_threshold <- as.numeric(summary_row$companion_threshold[[1]])
peak_net_benefit <- as.numeric(summary_row$peak_net_benefit[[1]])
peak_net_benefit_threshold <- as.numeric(summary_row$peak_net_benefit_threshold[[1]])
clinical_threshold_min <- as.numeric(summary_row$clinical_threshold_min[[1]])
clinical_threshold_max <- as.numeric(summary_row$clinical_threshold_max[[1]])
n_samples <- if ("n_samples" %in% names(summary_row)) as.numeric(summary_row$n_samples[[1]]) else NA_real_

curve_long <- rbind(
  data.frame(threshold = curve_df$threshold, strategy = "Winner panel", net_benefit = curve_df$winner_panel),
  data.frame(threshold = curve_df$threshold, strategy = "Treat all", net_benefit = curve_df$treat_all),
  data.frame(threshold = curve_df$threshold, strategy = "Treat none", net_benefit = curve_df$treat_none)
)

curve_long$strategy <- factor(
  curve_long$strategy,
  levels = c("Winner panel", "Treat all", "Treat none")
)
line_colors <- c(
  "Winner panel" = primary_color,
  "Treat all" = treat_all_color,
  "Treat none" = treat_none_color
)
line_types <- c(
  "Winner panel" = "solid",
  "Treat all" = "22",
  "Treat none" = "solid"
)
line_widths <- c(
  "Winner panel" = 1.35,
  "Treat all" = 0.9,
  "Treat none" = 0.9
)

annotation_label <- paste0(
  "Peak net benefit = ", sprintf("%.3f", peak_net_benefit),
  " at ", sprintf("p = %.2f", peak_net_benefit_threshold),
  "\nClinical window = ", sprintf("%.2f", clinical_threshold_min), "-", sprintf("%.2f", clinical_threshold_max),
  "\nScenario = ", scenario_name,
  if (!is.na(n_samples)) paste0("\nn = ", as.integer(n_samples)) else ""
)

y_lower <- -0.3
y_upper <- 0.5

annotation_x <- 0.60
annotation_y <- y_upper - 0.04 * (y_upper - y_lower)

figure_title <- if (nzchar(title_override)) {
  title_override
} else {
  "Decision Curve of the Winner Panel"
}

p <- ggplot(curve_long, aes(x = threshold, y = net_benefit, color = strategy, linetype = strategy)) +
  annotate(
    "rect",
    xmin = clinical_threshold_min,
    xmax = clinical_threshold_max,
    ymin = y_lower,
    ymax = y_upper,
    fill = primary_color,
    alpha = 0.06,
    color = NA
  ) +
  geom_line(aes(linewidth = strategy), lineend = "round") +
  scale_color_manual(values = line_colors, name = NULL) +
  scale_linetype_manual(values = line_types, name = NULL) +
  scale_linewidth_manual(values = line_widths, guide = "none") +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2),
    labels = function(x) sprintf("%.1f", x)
  ) +
  labs(
    title = figure_title,
    x = "Threshold probability",
    y = "Net benefit"
  ) +
  annotate(
    "label",
    x = 0.63,
    y = annotation_y,
    label = annotation_label,
    hjust = 0,
    vjust = 1,
    size = 3.35,
    fill = "white",
    color = text_color,
    label.padding = unit(0.20, "lines"),
    label.r = unit(0.12, "lines")
  ) +
  coord_cartesian(ylim = c(y_lower, y_upper), expand = FALSE) +
  theme_bw(base_family = font_family, base_size = 11) +
  theme(
    plot.background = element_rect(fill = "white", color = "white"),
    panel.background = element_rect(fill = "white", color = "white"),
    plot.title = element_text(face = "bold", size = 15.5, hjust = 0.5, color = text_color),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#374151"),
    panel.grid.major = element_line(color = grid_color, linewidth = 0.32),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#D7DEE8", linewidth = 0.7),
    legend.position = "top",
    legend.justification = "center",
    legend.box = "horizontal",
    legend.text = element_text(size = 10, color = "#374151"),
    legend.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(12, 16, 12, 12)
  )

save_plot(p, output_stem, width = 8.8, height = 6.8)
