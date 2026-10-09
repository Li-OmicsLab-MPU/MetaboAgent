#!/usr/bin/env Rscript

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

source(file.path(get_script_dir(), "font_family_helper.R"))

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript plot_phase2_incremental_value_summary.R <summary.json> <output_prefix> [theme_json]")
}

summary_path <- args[[1]]
output_prefix <- args[[2]]
theme_json <- ifelse(length(args) >= 3, args[[3]], "")
font_family <- "sans"
text_color <- "#1F2937"
axis_text_color <- "#374151"
grid_color <- "#E5E7EB"
panel_border_color <- "#D7DEE8"
winner_color <- "#6E3B33"
baseline_color <- "#6D9185"
uncertainty_fill <- "#D5DFDD"
neutral_color <- "#94A3B8"
if (nzchar(theme_json) && file.exists(theme_json)) {
  theme <- fromJSON(theme_json, simplifyVector = TRUE)
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- theme$font_family
  }
  if (!is.null(theme$text_color) && nzchar(theme$text_color)) text_color <- theme$text_color
  if (!is.null(theme$axis_text_color) && nzchar(theme$axis_text_color)) axis_text_color <- theme$axis_text_color
  if (!is.null(theme$grid_color) && nzchar(theme$grid_color)) grid_color <- theme$grid_color
  if (!is.null(theme$panel_border_color) && nzchar(theme$panel_border_color)) panel_border_color <- theme$panel_border_color
  if (!is.null(theme$semantic_colors$winner) && nzchar(theme$semantic_colors$winner)) winner_color <- theme$semantic_colors$winner
  if (!is.null(theme$semantic_colors$baseline) && nzchar(theme$semantic_colors$baseline)) baseline_color <- theme$semantic_colors$baseline
  if (!is.null(theme$semantic_colors$uncertainty_fill) && nzchar(theme$semantic_colors$uncertainty_fill)) uncertainty_fill <- theme$semantic_colors$uncertainty_fill
  if (!is.null(theme$semantic_colors$reference) && nzchar(theme$semantic_colors$reference)) neutral_color <- theme$semantic_colors$reference
}
font_family <- resolve_font_family(font_family)

payload <- fromJSON(summary_path, simplifyVector = TRUE)
summary <- payload$summary
if (is.null(summary) || identical(summary, list()) || !isTRUE(summary$enabled)) {
  stop("summary.json does not contain an enabled incremental-value payload")
}

nri <- summary$nri
idi <- summary$idi
delta_risk <- summary$delta_risk_summary

plot_df <- data.frame(
  metric = factor(c("Continuous NRI", "IDI"), levels = c("Continuous NRI", "IDI")),
  estimate = c(as.numeric(nri$value), as.numeric(idi$value)),
  ci_low = c(as.numeric(nri$ci_95[[1]]), as.numeric(idi$ci_95[[1]])),
  ci_high = c(as.numeric(nri$ci_95[[2]]), as.numeric(idi$ci_95[[2]])),
  fill = c(winner_color, baseline_color),
  label = c(
    sprintf(
      "%.3f  [%.3f, %.3f]  p = %.3f",
      as.numeric(nri$value), as.numeric(nri$ci_95[[1]]), as.numeric(nri$ci_95[[2]]), as.numeric(nri$p_value_empirical)
    ),
    sprintf(
      "%.3f  [%.3f, %.3f]  p = %.3f",
      as.numeric(idi$value), as.numeric(idi$ci_95[[1]]), as.numeric(idi$ci_95[[2]]), as.numeric(idi$p_value_empirical)
    )
  ),
  stringsAsFactors = FALSE
)

x_min <- min(c(plot_df$ci_low, 0))
x_max <- max(c(plot_df$ci_high, 0))
x_span <- x_max - x_min
if (!is.finite(x_span) || x_span <= 0) {
  x_span <- 0.1
}
x_left <- x_min - 0.08 * x_span
x_right <- x_max + 0.50 * x_span
left_breaks <- c(-0.30, 0.00, 0.30)
left_breaks <- left_breaks[left_breaks >= x_left & left_breaks <= x_right]
label_panel_xmin <- x_max + 0.03 * x_span
label_panel_xmax <- x_right
label_text_x <- x_max + 0.07 * x_span

annotation_label <- sprintf(
  "Event component = %.3f   |   Non-event component = %.3f   |   Mean delta risk (events / nonevents) = %.3f / %.3f",
  as.numeric(nri$events_component),
  as.numeric(nri$nonevents_component),
  as.numeric(delta_risk$mean_delta_events),
  as.numeric(delta_risk$mean_delta_nonevents)
)

p <- ggplot(plot_df, aes(y = metric, x = estimate)) +
  geom_vline(xintercept = left_breaks[left_breaks != 0], color = grid_color, linewidth = 0.5) +
  geom_vline(xintercept = 0, color = neutral_color, linetype = "solid", linewidth = 0.95) +
  geom_segment(
    aes(x = ci_low, xend = ci_high, y = metric, yend = metric),
    linewidth = 1.15,
    color = neutral_color,
    lineend = "round"
  ) +
  geom_segment(
    aes(x = ci_low, xend = ci_low, y = as.numeric(metric) - 0.10, yend = as.numeric(metric) + 0.10),
    linewidth = 0.75,
    color = neutral_color
  ) +
  geom_segment(
    aes(x = ci_high, xend = ci_high, y = as.numeric(metric) - 0.10, yend = as.numeric(metric) + 0.10),
    linewidth = 0.75,
    color = neutral_color
  ) +
  geom_point(
    aes(fill = fill),
    shape = 21,
    size = 4.8,
    stroke = 0.6,
    color = text_color,
    show.legend = FALSE
  ) +
  annotate(
    "rect",
    xmin = label_panel_xmin,
    xmax = label_panel_xmax,
    ymin = 0.45,
    ymax = 2.55,
    fill = alpha(uncertainty_fill, 0.18),
    color = NA
  ) +
  geom_text(
    aes(x = label_text_x, label = label),
    hjust = 0,
    size = 4.0,
    family = font_family,
    color = text_color
  ) +
  annotate(
    "text",
    x = mean(c(x_min, x_max)),
    y = 2.19,
    hjust = 0.5,
    size = 3.35,
    family = font_family,
    color = axis_text_color,
    label = annotation_label
  ) +
  scale_fill_identity() +
  scale_x_continuous(
    limits = c(x_left, x_right),
    expand = c(0, 0),
    breaks = left_breaks,
    labels = function(x) sprintf("%.2f", x)
  ) +
  coord_cartesian(clip = "off") +
  labs(
    title = payload$figure_title,
    x = payload$x_label,
    y = NULL
  ) +
  theme_minimal(base_family = font_family, base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.y = element_text(size = 12, color = text_color, face = "bold"),
    axis.text.x = element_text(size = 10.5, color = axis_text_color),
    axis.title.x = element_text(size = 11.2, color = text_color, face = "bold"),
    plot.title = element_text(size = 15, face = "bold", hjust = 0.5, color = text_color),
    plot.subtitle = element_blank(),
    plot.caption = element_blank(),
    plot.margin = margin(16, 116, 16, 24)
  )

png_path <- paste0(output_prefix, ".png")
pdf_path <- paste0(output_prefix, ".pdf")
svg_path <- paste0(output_prefix, ".svg")

ggsave(png_path, plot = p, width = 12.4, height = 5.6, dpi = 320, bg = "white", type = "cairo")
ggsave(pdf_path, plot = p, width = 12.4, height = 5.6, bg = "white", device = cairo_pdf, family = font_family)
if (requireNamespace("svglite", quietly = TRUE)) {
  ggsave(svg_path, plot = p, width = 12.4, height = 5.6, bg = "white")
}

cat(png_path, "\n")
cat(pdf_path, "\n")
if (file.exists(svg_path)) {
  cat(svg_path, "\n")
}
