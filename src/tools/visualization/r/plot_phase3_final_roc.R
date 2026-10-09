args <- commandArgs(trailingOnly = TRUE)

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

if (length(args) < 3) {
  stop("Usage: Rscript plot_phase3_final_roc.R <roc_csv> <summary_csv> <output_stem> [theme_json] [curve_mode] [title]")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
})

primary_color <- "#8C1D18"
primary_fill <- "#D9A5A0"
neutral_line <- "#B8C0CC"
text_color <- "#1F2937"
grid_color <- "#E9EDF2"

roc_csv <- args[[1]]
summary_csv <- args[[2]]
output_stem <- args[[3]]
theme_json <- ifelse(length(args) >= 4, args[[4]], "")
curve_mode <- ifelse(length(args) >= 5, args[[5]], "cv")
title_override <- ifelse(length(args) >= 6, args[[6]], "")
font_family <- "sans"
# Both development OOF and internal holdout curves receive their own
# bootstrap uncertainty interval from the canonical audit predictions.
show_ci <- TRUE

if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  primary_color <- theme$semantic_colors$winner
  primary_fill <- theme$primary_fill
  neutral_line <- theme$semantic_colors$reference
  text_color <- theme$text_color
  grid_color <- theme$grid_color
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- theme$font_family
  }
}

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

roc_df <- read.csv(roc_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_df <- read.csv(summary_csv, check.names = FALSE, stringsAsFactors = FALSE)
summary_row <- summary_df[1, , drop = FALSE]

feature_count <- as.integer(summary_row$feature_count[[1]])
pooled_auc <- as.numeric(summary_row$pooled_auc[[1]])
pooled_auc_ci_low <- as.numeric(summary_row$pooled_auc_ci_low[[1]])
pooled_auc_ci_high <- as.numeric(summary_row$pooled_auc_ci_high[[1]])

auc_label <- if (show_ci) {
  sprintf(
    "AUC = %.3f (95%% CI: %.3f-%.3f)",
    pooled_auc,
    pooled_auc_ci_low,
    pooled_auc_ci_high
  )
} else {
  sprintf("AUC = %.3f", pooled_auc)
}

title_text <- if (nzchar(title_override)) {
  title_override
} else if (curve_mode == "holdout") {
  "ROC Curve of the Winner Panel on Internal Holdout"
} else {
  "ROC Curve of the Winner Panel under Internal Cross-validation"
}

p <- ggplot() +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "22",
    linewidth = 0.70,
    color = neutral_line
  )

if (show_ci) {
  p <- p +
    geom_ribbon(
      data = roc_df,
      aes(x = fpr, ymin = tpr_lower, ymax = tpr_upper),
      fill = primary_fill,
      alpha = 0.38
    )
}

p <- p +
  geom_path(
    data = roc_df,
    aes(x = fpr, y = tpr),
    color = primary_color,
    linewidth = 1.7,
    lineend = "round"
  ) +
  annotate(
    "label",
    x = 0.05,
    y = 0.94,
    label = auc_label,
    hjust = 0,
    vjust = 1,
    size = 3.7,
    fill = "white",
    color = text_color,
    label.padding = unit(0.20, "lines"),
    label.r = unit(0.12, "lines")
  ) +
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
    title = title_text,
    x = "False positive rate (1 - specificity)",
    y = "True positive rate (sensitivity)"
  ) +
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
    plot.margin = margin(12, 16, 12, 12)
  )

save_plot(p, output_stem, width = 8.2, height = 6.8)
