args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 6) {
  stop("Usage: Rscript plot_phase1_corr_heatmap.R <corr_csv> <stars_csv> <output_stem> <method> <cut_n> <title> [theme_json]")
}

source(file.path(getwd(), "src/tools/visualization/r/font_family_helper.R"))

suppressPackageStartupMessages({
  library(pheatmap)
  library(RColorBrewer)
  library(grid)
})

corr_csv <- args[[1]]
stars_csv <- args[[2]]
output_stem <- args[[3]]
cluster_method <- args[[4]]
cut_n <- as.integer(args[[5]])
plot_title <- args[[6]]
theme_json <- ifelse(length(args) >= 7, args[[7]], "")

negative_color <- "#202754"
zero_color <- "#FFFFFF"
positive_color <- "#E96362"
text_color <- "#1F2937"
panel_border_color <- "#D7DEE8"
font_family <- "sans"
if (nzchar(theme_json) && file.exists(theme_json)) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    lib_candidates <- c(Sys.getenv("METABOAGENT_R_LIB"), Sys.getenv("R_LIBS_USER"), file.path(getwd(), ".r_libs"))
    lib_candidates <- lib_candidates[nzchar(lib_candidates)]
    lib_candidates <- lib_candidates[file.exists(lib_candidates)]
    if (length(lib_candidates) > 0) {
      .libPaths(unique(c(lib_candidates, .libPaths())))
    }
  }
  suppressPackageStartupMessages(library(jsonlite))
  theme <- jsonlite::fromJSON(theme_json)
  negative_color <- theme$semantic_colors$baseline
  zero_color <- theme$zero_value_color
  positive_color <- theme$semantic_colors$winner
  text_color <- theme$text_color
  panel_border_color <- theme$panel_border_color
  if (!is.null(theme$font_family) && nzchar(theme$font_family)) {
    font_family <- theme$font_family
  }
}
font_family <- resolve_font_family(font_family)

corr_df <- read.csv(corr_csv, row.names = 1, check.names = FALSE)
stars_df <- read.csv(
  stars_csv,
  row.names = 1,
  check.names = FALSE,
  na.strings = NULL,
  strip.white = FALSE
)
stars_df[is.na(stars_df)] <- ""
stars_df[trimws(stars_df) == "NA"] <- ""

corr_mat <- as.matrix(corr_df)
stars_mat <- as.matrix(stars_df)
mode(corr_mat) <- "numeric"
mode(stars_mat) <- "character"

# Use signed correlation distance so similarly colored blocks cluster together.
# r = 1  -> distance 0
# r = 0  -> distance 0.5
# r = -1 -> distance 1
distance_mat <- as.dist((1 - corr_mat) / 2)
hc <- hclust(distance_mat, method = cluster_method)

palette_fn <- colorRampPalette(c(negative_color, zero_color, positive_color))
heat_colors <- palette_fn(100)
breaks <- seq(-1, 1, length.out = 101)

render_heatmap <- function() {
  pheatmap(
    corr_mat,
    cluster_rows = hc,
    cluster_cols = hc,
    clustering_method = cluster_method,
    cutree_rows = cut_n,
    cutree_cols = cut_n,
    color = heat_colors,
    breaks = breaks,
    display_numbers = stars_mat,
    number_color = text_color,
    fontsize_number = 8,
    cellwidth = 16,
    cellheight = 16,
    angle_col = 90,
    border_color = panel_border_color,
    legend = TRUE,
    legend_breaks = c(-1, -0.5, 0, 0.5, 1),
    treeheight_row = 55,
    treeheight_col = 55,
    fontsize_row = 11,
    fontsize_col = 11,
    silent = TRUE,
    main = plot_title
  )
}

save_plot <- function(file_path, device_type) {
  if (device_type == "pdf") {
    cairo_pdf(file_path, width = 10.8, height = 9.4, family = font_family, bg = "white")
  } else if (device_type == "png") {
    png(file_path, width = 2700, height = 2350, res = 300, type = "cairo", bg = "white")
  } else if (device_type == "svg") {
    svg(file_path, width = 10.8, height = 9.4, bg = "white")
  } else {
    stop(sprintf("Unsupported device type: %s", device_type))
  }
  hm <- render_heatmap()
  grid.newpage()
  grid.draw(hm$gtable)
  dev.off()
}

save_plot(paste0(output_stem, ".pdf"), "pdf")
save_plot(paste0(output_stem, ".png"), "png")
save_plot(paste0(output_stem, ".svg"), "svg")
