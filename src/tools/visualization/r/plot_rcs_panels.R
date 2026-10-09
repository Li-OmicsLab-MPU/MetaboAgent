#!/usr/bin/env Rscript

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

user_lib_candidates <- c(
  Sys.getenv("METABOAGENT_R_LIB", unset = ""),
  Sys.getenv("R_LIBS_USER"),
  Sys.glob(path.expand("~/R/*-library/*"))
)
.libPaths(unique(c(user_lib_candidates[nzchar(user_lib_candidates)], .libPaths())))

suppressPackageStartupMessages({
  library(jsonlite)
  library(plotRCS)
  library(ggplot2)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript plot_rcs_panels.R <payload.json>")
}

payload <- fromJSON(args[[1]])
font_family <- if (!is.null(payload$font_family) && nzchar(payload$font_family)) payload$font_family else "Times"
source(file.path(get_script_dir(), "font_family_helper.R"))
font_family <- resolve_font_family(font_family)

plot_failure_panel <- function(display_name, message_text, fontsize = 10) {
  ggplot() +
    annotate(
      "text",
      x = 0.5,
      y = 0.5,
      label = paste(display_name, message_text, sep = "\n"),
      family = font_family,
      size = fontsize / 3
    ) +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_void()
}

format_p_value <- function(p_value) {
  if (is.null(p_value) || length(p_value) == 0 || is.na(p_value)) {
    return("NA")
  }
  if (p_value < 0.001) {
    return("< 0.001")
  }
  sprintf("= %.3f", p_value)
}

fallback_knots <- function(x) {
  unique_x <- sort(unique(x))
  if (length(unique_x) < 3) {
    return(rep(NA_real_, 3))
  }
  idx <- c(
    max(1, floor(0.10 * (length(unique_x) - 1)) + 1),
    round(0.50 * (length(unique_x) - 1)) + 1,
    min(length(unique_x), ceiling(0.90 * (length(unique_x) - 1)) + 1)
  )
  as.numeric(unique_x[idx])
}

data_path <- payload$data_path
target_column <- payload$target_column
features <- unlist(payload$features)
display_features <- if (!is.null(payload$display_features)) unlist(payload$display_features) else features
save_path <- payload$save_path
summary_path <- payload$summary_path
n_grid <- as.integer(payload$n_grid)
figsize <- as.numeric(unlist(payload$figsize))
line_color <- payload$line_color
scatter_alpha <- as.numeric(payload$scatter_alpha)
fontsize <- max(as.numeric(payload$fontsize) + 1, 12)
title_fontsize <- max(as.numeric(payload$title_fontsize) + 2, 16)
label_fontsize <- max(as.numeric(payload$label_fontsize) + 1, 14)

df <- read.csv(data_path, check.names = FALSE, stringsAsFactors = FALSE)
required_columns <- c(features, target_column)
missing_columns <- setdiff(required_columns, names(df))
if (length(missing_columns) > 0) {
  stop(sprintf("Columns not found in data: %s", paste(missing_columns, collapse = ", ")))
}

y_raw <- df[[target_column]]
class_counts <- table(y_raw)
positive_class <- names(class_counts)[which.min(class_counts)][1]
y_pos_full <- as.integer(as.character(y_raw) == positive_class)
prevalence <- mean(y_pos_full, na.rm = TRUE)

n_feat <- length(features)
n_cols <- if (n_feat > 1) 2 else 1
n_rows <- ceiling(n_feat / n_cols)

dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
plot_list <- vector("list", n_feat)
summary_rows <- vector("list", n_feat)

for (i in seq_along(features)) {
  feature <- features[[i]]
  display_name <- display_features[[i]]
  x_raw <- suppressWarnings(as.numeric(df[[feature]]))
  mask <- is.finite(x_raw) & is.finite(y_pos_full)
  x <- x_raw[mask]
  y <- y_pos_full[mask]

  if (length(unique(x)) < 4) {
    plot_list[[i]] <- plot_failure_panel(display_name, "Insufficient unique values", fontsize)
    summary_rows[[i]] <- list(
      feature = feature,
      display_feature = display_name,
      status = "failed",
      reason = "Insufficient unique values"
    )
    next
  }

  lo <- as.numeric(quantile(x, 0.05, na.rm = TRUE, type = 7))
  hi <- as.numeric(quantile(x, 0.95, na.rm = TRUE, type = 7))
  if (!is.finite(lo) || !is.finite(hi) || hi <= lo) {
    plot_list[[i]] <- plot_failure_panel(display_name, "Insufficient spread for core-range prediction", fontsize)
    summary_rows[[i]] <- list(
      feature = feature,
      display_feature = display_name,
      status = "failed",
      reason = "Insufficient spread for core-range prediction"
    )
    next
  }

  knot_positions <- as.numeric(quantile(x, c(0.10, 0.50, 0.90), na.rm = TRUE, type = 7))
  if (any(!is.finite(knot_positions)) || any(diff(knot_positions) <= 0)) {
    knot_positions <- fallback_knots(x)
  }
  if (any(!is.finite(knot_positions)) || any(diff(knot_positions) <= 0)) {
    plot_list[[i]] <- plot_failure_panel(display_name, "Unable to place 3 stable RCS knots", fontsize)
    summary_rows[[i]] <- list(
      feature = feature,
      display_feature = display_name,
      status = "failed",
      reason = "Unable to place 3 stable RCS knots"
    )
    next
  }

  boundary_lo <- as.numeric(knot_positions[1])
  internal_knot <- as.numeric(knot_positions[2])
  boundary_hi <- as.numeric(knot_positions[3])
  median_val <- as.numeric(median(x, na.rm = TRUE))
  if (!(boundary_lo < internal_knot && internal_knot < boundary_hi)) {
    plot_list[[i]] <- plot_failure_panel(display_name, "Unable to place ordered boundary/internal knots", fontsize)
    summary_rows[[i]] <- list(
      feature = feature,
      display_feature = display_name,
      status = "failed",
      reason = "Unable to place ordered boundary/internal knots"
    )
    next
  }

  fit_warning_count <- 0L
  fit_warning_msgs <- character(0)
  p_nonlin <- NA_real_
  p_overall <- NA_real_
  xbreaks_core <- pretty(c(lo, hi))
  ybreaks_or <- pretty(c(0, 5))
  x_range <- hi - lo
  display_xmin <- lo
  display_xmax <- hi
  p_label_x <- hi - 0.01 * x_range
  p_label_y <- 4.85

  withCallingHandlers(
    {
      df_model <- data.frame(
        outcome = factor(y, levels = c(0, 1), labels = c("0", "1")),
        exposure = x
      )
      dd_name <- sprintf("ddist_manual_%d", i)
      old_datadist <- getOption("datadist")
      assign(dd_name, rms::datadist(df_model), envir = .GlobalEnv)
      options(datadist = dd_name)
      model_manual <- rms::lrm(
        outcome ~ rms::rcs(exposure, c(boundary_lo, internal_knot, boundary_hi)),
        data = df_model,
        x = TRUE,
        y = TRUE
      )
      anova_table <- as.data.frame(anova(model_manual))
      row_names <- trimws(rownames(anova_table))
      if ("exposure" %in% row_names && "P" %in% colnames(anova_table)) {
        p_overall <- as.numeric(anova_table[match("exposure", row_names), "P"])
      } else if (nrow(anova_table) >= 1 && "P" %in% colnames(anova_table)) {
        p_overall <- as.numeric(anova_table[1, "P"])
      }
      if ("Nonlinear" %in% row_names && "P" %in% colnames(anova_table)) {
        p_nonlin <- as.numeric(anova_table[match("Nonlinear", row_names), "P"])
      } else if (nrow(anova_table) >= 2 && "P" %in% colnames(anova_table)) {
        p_nonlin <- as.numeric(anova_table[2, "P"])
      }
      options(datadist = old_datadist)
      if (exists(dd_name, envir = .GlobalEnv, inherits = FALSE)) {
        rm(list = dd_name, envir = .GlobalEnv)
      }
      plot_obj <- rcsplot(
        data = df_model,
        outcome = "outcome",
        exposure = "exposure",
        positive = "1",
        knots = c(0.10, 0.50, 0.90),
        ref.value = median_val,
        knots.line = FALSE,
        ref.line = TRUE,
        conf.int = TRUE,
        conf.level = 0.95,
        conf.type = "shape",
        pvalue = FALSE,
        pvalue.position = c(0.02, 0.98),
        pvalue.label.overall = "P for overall",
        pvalue.label.nonlinear = "P for non-linearity",
        fontsize = fontsize,
        fontfamily = font_family,
        linecolor = line_color,
        alpha = 0.2,
        xbreaks = xbreaks_core,
        ybreaks = ybreaks_or,
        xlab = display_name,
        ylab = "",
        explain = FALSE
      )
    },
    warning = function(w) {
      warning_msg <- conditionMessage(w)
      is_known_plotrcs_deprecation <- grepl("aes_string\\(\\)|`size` argument", warning_msg)
      if (!is_known_plotrcs_deprecation) {
        fit_warning_msgs <<- c(fit_warning_msgs, warning_msg)
        fit_warning_count <<- fit_warning_count + 1L
      }
      invokeRestart("muffleWarning")
    }
  )
  plot_data <- plot_obj$data
  plot_obj$coordinates <- coord_cartesian(xlim = c(display_xmin, display_xmax), ylim = c(0, 5), clip = "off")
  plot_obj$scales$scales <- Filter(
    function(scale_obj) !("x" %in% scale_obj$aesthetics),
    plot_obj$scales$scales
  )
  plot_obj <- suppressWarnings(
    plot_obj +
      scale_x_continuous(
        breaks = xbreaks_core,
        expand = expansion(mult = c(0.02, 0.02))
      ) +
      ggtitle(display_name) +
      annotate(
        "text",
        x = p_label_x,
        y = p_label_y,
        label = paste0(
          "P for overall ", format_p_value(p_overall), "\n",
          "P for non-linearity ", format_p_value(p_nonlin)
        ),
        hjust = 1,
        vjust = 1,
        family = font_family,
        size = fontsize / 3,
        lineheight = 1.1
      ) +
      theme(
        text = element_text(family = font_family, size = fontsize),
        plot.title = element_text(size = title_fontsize, face = "bold"),
        axis.title = element_text(size = label_fontsize),
        axis.text = element_text(size = fontsize),
        panel.grid.minor = element_blank(),
        plot.margin = margin(t = 12, r = 16, b = 12, l = 12)
      )
  )
  plot_list[[i]] <- plot_obj

  summary_rows[[i]] <- list(
    feature = feature,
    display_feature = display_name,
    status = "completed",
    p_overall = p_overall,
    p_nonlinearity = p_nonlin,
    n_samples = length(x),
    effective_n_knots = 3,
    boundary_knot_p10 = boundary_lo,
    internal_knot_p50 = internal_knot,
    boundary_knot_p90 = boundary_hi,
    reference_value_median = median_val,
    x_plot_min_p05 = lo,
    x_plot_max_p95 = hi,
    y_plot_min = if ("lower" %in% names(plot_data)) min(plot_data$lower, na.rm = TRUE) else NA_real_,
    y_plot_max = if ("upper" %in% names(plot_data)) max(plot_data$upper, na.rm = TRUE) else NA_real_,
    y_display_min = 0,
    y_display_max = 5,
    fit_warning_count = fit_warning_count,
    fit_warning_messages = fit_warning_msgs
  )
}

if (n_feat < (n_rows * n_cols)) {
  for (j in seq.int(n_feat + 1, n_rows * n_cols)) {
    plot_list[[j]] <- plot_failure_panel("", "", fontsize)
  }
}

title_text <- "Restricted Cubic Spline Risk Curves of the Winner Panel"

pdf(
  file = save_path,
  width = figsize[1],
  height = figsize[2],
  family = font_family,
  pointsize = fontsize,
  onefile = TRUE
)
on.exit(dev.off(), add = TRUE)

grid.newpage()
pushViewport(viewport(layout = grid.layout(n_rows + 1, n_cols, heights = unit.c(unit(0.45, "in"), rep(unit(1, "null"), n_rows)))))
grid.text(title_text, vp = viewport(layout.pos.row = 1, layout.pos.col = seq_len(n_cols)), gp = gpar(fontface = "bold", fontsize = title_fontsize, col = "#111827", fontfamily = font_family))
pushViewport(viewport(layout.pos.row = 2:(n_rows + 1), layout.pos.col = seq_len(n_cols), x = 0.5, y = 0.5, width = 0.98, height = 0.98, layout = grid.layout(n_rows, n_cols)))
for (idx in seq_len(n_rows * n_cols)) {
  row_idx <- ((idx - 1) %/% n_cols) + 1
  col_idx <- ((idx - 1) %% n_cols) + 1
  if (!is.null(plot_list[[idx]])) {
    print(plot_list[[idx]], vp = viewport(layout.pos.row = row_idx, layout.pos.col = col_idx))
  }
}

summary_payload <- list(
  positive_class = positive_class,
  prevalence = prevalence,
  n_features = n_feat,
  backend = "r",
  engine = "plotRCS",
  features = summary_rows
)

write_json(summary_payload, summary_path, pretty = TRUE, auto_unbox = TRUE, null = "null")

message(sprintf("R RCS panel plot saved to: %s", save_path))
message(sprintf("R RCS summary saved to: %s", summary_path))
