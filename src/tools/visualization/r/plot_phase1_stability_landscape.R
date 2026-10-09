args <- commandArgs(trailingOnly = TRUE)

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

if (length(args) < 3) {
  stop("Usage: Rscript plot_phase1_stability_landscape.R <csv_path> <output_stem> <title> [theme_json]")
}

local_lib <- Sys.getenv("METABOAGENT_R_LIB", unset = "")
if (nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}
source(file.path(get_script_dir(), "font_family_helper.R"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(grid)
})

csv_path <- args[[1]]
output_stem <- args[[2]]
plot_title <- args[[3]]
theme_json <- ifelse(length(args) >= 4, args[[4]], "")
font_family <- resolve_font_family("sans")

text_color <- "#1F2937"
axis_text_color <- "#374151"
grid_color <- "#E5E7EB"
reference_color <- "#BBBDC0"
uncertainty_fill <- "#F2F4F7"
highlight_color <- "#202754"
status_colors <- c(
  "mandatory" = "#FCC248",
  "stable_core" = "#3D85C6",
  "completed" = "#19AA9D",
  "rejected" = "#BBBDC0"
)
if (nzchar(theme_json) && file.exists(theme_json)) {
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  text_color <- theme$text_color
  axis_text_color <- theme$axis_text_color
  grid_color <- theme$grid_color
  reference_color <- theme$semantic_colors$reference
  uncertainty_fill <- theme$primary_fill
  highlight_color <- theme$semantic_colors$highlight
  status_colors <- c(
    "mandatory" = theme$status_colors$mandatory,
    "stable_core" = theme$status_colors$stable_core,
    "completed" = theme$status_colors$completed,
    "rejected" = theme$status_colors$rejected
  )
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- theme$font_family
  }
}
font_family <- resolve_font_family(font_family)

df <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)
df$is_final_panel <- df$is_final_panel %in% c(TRUE, "TRUE", "True", 1, "1")

wrap_feature_label <- function(x, width = 24) {
  if (!nzchar(x)) return(x)
  x <- gsub(" / ", " /\n", x, fixed = TRUE)
  if (grepl("\n", x, fixed = TRUE)) return(x)
  parts <- strwrap(x, width = width)
  paste(parts, collapse = "\n")
}

df$display_label_wrapped <- vapply(df$display_name, wrap_feature_label, character(1), USE.NAMES = FALSE)

df$status <- factor(
  df$status,
  levels = c("rejected", "completed", "stable_core", "mandatory")
)

legend_labels <- c(
  "mandatory" = "Mandatory",
  "stable_core" = "Stable Core",
  "completed" = "Greedy Completed",
  "rejected" = "Rejected"
)

label_df <- df[df$is_final_panel, , drop = FALSE]

p <- ggplot(df, aes(x = selection_frequency, y = method_consensus)) +
  annotate("rect", xmin = 0.60, xmax = 1.02, ymin = 3, ymax = 7.4,
           fill = uncertainty_fill, alpha = 0.82) +
  annotate("rect", xmin = 0.0, xmax = 0.60, ymin = 0.0, ymax = 3,
           fill = "#F8FAFC", alpha = 0.98) +
  geom_vline(xintercept = 0.60, color = reference_color, linewidth = 0.55, linetype = "dashed") +
  geom_hline(yintercept = 3, color = reference_color, linewidth = 0.55, linetype = "dashed") +
  geom_point(
    aes(size = panel_score, fill = status, alpha = status),
    shape = 21, color = "white", stroke = 0.35,
    position = position_jitter(width = 0.008, height = 0.045, seed = 42)
  ) +
  ggrepel::geom_label_repel(
    data = label_df,
    aes(label = display_label_wrapped),
    size = 2.7,
    label.size = 0,
    fill = scales::alpha("white", 0.88),
    color = text_color,
    box.padding = 0.42,
    point.padding = 0.22,
    segment.color = reference_color,
    segment.alpha = 0.65,
    segment.size = 0.3,
    min.segment.length = 0,
    max.overlaps = 20,
    seed = 42
  ) +
  annotate("text", x = 0.82, y = 5.95, label = "Stable Core Zone",
           color = highlight_color, alpha = 0.90, size = 3.7, fontface = "italic") +
  annotate("text", x = 0.18, y = 1.45, label = "Low Frequency\nLow Consensus",
           color = axis_text_color, size = 3.2) +
  annotate("text", x = 0.83, y = 1.45, label = "High Frequency\nLow Consensus",
           color = axis_text_color, size = 3.2) +
  annotate("text", x = 0.18, y = 5.7, label = "Low Frequency\nHigh Consensus",
           color = axis_text_color, size = 3.2) +
  scale_fill_manual(values = status_colors, breaks = names(legend_labels), labels = legend_labels) +
  scale_alpha_manual(values = c("mandatory" = 0.96, "stable_core" = 0.94, "completed" = 0.90, "rejected" = 0.40), guide = "none") +
  scale_size_continuous(range = c(2.8, 12.5), name = "Panel score") +
  scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
  scale_y_continuous(breaks = 0:7, expand = c(0, 0)) +
  coord_cartesian(xlim = c(-0.02, 1.02), ylim = c(0.0, 7.4), clip = "off") +
  labs(
    title = plot_title,
    x = "Selection Frequency",
    y = "Method Consensus",
    fill = "Feature Status"
  ) +
  guides(
    fill = guide_legend(order = 1, override.aes = list(size = 5, alpha = 1)),
    size = guide_legend(order = 2, override.aes = list(fill = "#A3A3A3", color = "#4B5563", alpha = 1, shape = 21, stroke = 0.4))
  ) +
  theme_bw(base_family = font_family, base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 10)),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10, color = axis_text_color),
    panel.grid.major = element_line(color = grid_color, linewidth = 0.35),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    legend.position = "right",
    legend.key = element_rect(fill = "white", colour = NA),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    plot.margin = margin(12, 26, 18, 10)
  )

save_plot <- function(file_path, device_type) {
  if (device_type == "pdf") {
    pdf(file_path, width = 10.8, height = 8.2, useDingbats = FALSE)
  } else if (device_type == "png") {
    png(file_path, width = 3240, height = 2460, res = 300)
  } else if (device_type == "svg") {
    svg(file_path, width = 10.8, height = 8.2)
  } else {
    stop(sprintf("Unsupported device type: %s", device_type))
  }
  print(p)
  dev.off()
}

save_plot(paste0(output_stem, ".pdf"), "pdf")
save_plot(paste0(output_stem, ".png"), "png")
save_plot(paste0(output_stem, ".svg"), "svg")
