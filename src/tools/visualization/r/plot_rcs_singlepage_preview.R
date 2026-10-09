#!/usr/bin/env Rscript

user_lib_candidates <- c(
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
  stop("Usage: Rscript plot_rcs_singlepage_preview.R <payload.json>")
}

payload <- fromJSON(args[[1]])
text_color <- if (!is.null(payload$text_color) && nzchar(payload$text_color)) payload$text_color else "#1F2937"
axis_text_color <- if (!is.null(payload$axis_text_color) && nzchar(payload$axis_text_color)) payload$axis_text_color else "#374151"
grid_color <- if (!is.null(payload$grid_color) && nzchar(payload$grid_color)) payload$grid_color else "#E5E7EB"
panel_border_color <- if (!is.null(payload$panel_border_color) && nzchar(payload$panel_border_color)) payload$panel_border_color else "#D1D5DB"

format_p_value <- function(p_value) {
  if (is.null(p_value) || length(p_value) == 0 || is.na(p_value)) {
    return("NA")
  }
  if (p_value < 0.001) {
    return("< 0.001")
  }
  sprintf("= %.3f", p_value)
}

normalize_display_name <- function(label) {
  value <- trimws(as.character(label))
  mapping <- c(
    "2_DEOXYURIDINE" = "2-Deoxyuridine",
    "glucose" = "Glucose",
    "glycine" = "Glycine"
  )
  if (value %in% names(mapping)) {
    return(unname(mapping[[value]]))
  }
  value
}

format_axis_value <- function(value) {
  if (!is.finite(value)) {
    return("NA")
  }
  formatted <- if (abs(value) >= 100) {
    sprintf("%.0f", value)
  } else if (abs(value) >= 10) {
    sprintf("%.1f", value)
  } else if (abs(value) >= 1) {
    sprintf("%.2f", value)
  } else {
    sprintf("%.3f", value)
  }
  formatted <- sub("0+$", "", formatted)
  sub("\\.$", "", formatted)
}

compute_sample_skewness <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 3) {
    return(NA_real_)
  }
  mu <- mean(x)
  sigma <- stats::sd(x)
  if (!is.finite(sigma) || sigma <= 0) {
    return(NA_real_)
  }
  mean(((x - mu) / sigma) ^ 3)
}

run_shapiro_safely <- function(x) {
  x <- x[is.finite(x)]
  if (length(unique(x)) < 3) {
    return(NA_real_)
  }
  sample_n <- min(length(x), 5000)
  if (sample_n < 3) {
    return(NA_real_)
  }
  suppressWarnings(stats::shapiro.test(x[seq_len(sample_n)])$p.value)
}

should_apply_log_transform <- function(x, shapiro_p, skewness_value) {
  if (!all(is.finite(c(shapiro_p, skewness_value)))) {
    return(FALSE)
  }
  is_right_skewed <- skewness_value > 1
  failed_normality <- shapiro_p < 0.05
  has_positive_support <- all(x > 0, na.rm = TRUE)
  is_right_skewed && failed_normality && has_positive_support
}

find_or1_crossing <- function(plot_data, ref_x) {
  if (!all(c("exposure", "yhat") %in% names(plot_data))) {
    return(list(x = NA_real_, kind = "none"))
  }
  curve_df <- plot_data[is.finite(plot_data$exposure) & is.finite(plot_data$yhat), c("exposure", "yhat")]
  if (nrow(curve_df) < 3) {
    return(list(x = NA_real_, kind = "none"))
  }
  x <- as.numeric(curve_df$exposure)
  y_centered <- as.numeric(curve_df$yhat) - 1

  exact_idx <- which(abs(y_centered) < 1e-8)
  candidate_x <- numeric(0)
  if (length(exact_idx) > 0) {
    candidate_x <- c(candidate_x, x[exact_idx])
  }

  sign_change_idx <- which(y_centered[-length(y_centered)] * y_centered[-1] < 0)
  if (length(sign_change_idx) > 0) {
    interpolated <- vapply(
      sign_change_idx,
      function(idx) {
        approx(
          x = c(y_centered[idx], y_centered[idx + 1]),
          y = c(x[idx], x[idx + 1]),
          xout = 0
        )$y
      },
      numeric(1)
    )
    candidate_x <- c(candidate_x, interpolated)
  }

  candidate_x <- candidate_x[is.finite(candidate_x)]
  if (length(candidate_x) == 0) {
    return(list(x = NA_real_, kind = "none"))
  }
  chosen_x <- candidate_x[[which.min(abs(candidate_x - ref_x))]]
  list(x = as.numeric(chosen_x), kind = "or1_crossing")
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

plot_failure_panel <- function(display_name, message_text, fontsize = 10) {
  ggplot() +
    annotate(
      "text",
      x = 0.5,
      y = 0.5,
      label = paste(display_name, message_text, sep = "\n"),
      family = "Times",
      size = fontsize / 3,
      colour = "#4B5563"
    ) +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_void()
}

choose_layout <- function(n_feat) {
  if (n_feat <= 1) {
    return(c(1, 1))
  }
  if (n_feat <= 4) {
    return(c(2, ceiling(n_feat / 2)))
  }
  if (n_feat <= 9) {
    return(c(3, ceiling(n_feat / 3)))
  }
  c(4, ceiling(n_feat / 4))
}

render_page <- function(plot_list, n_rows, n_cols, title_text, subtitle_text, footnote_text, title_fontsize, subtitle_fontsize) {
  grid.newpage()
  grid.text(
    title_text,
    x = 0.5,
    y = unit(0.985, "npc"),
    gp = gpar(fontfamily = "Times", fontsize = title_fontsize, fontface = "bold", col = "#111827")
  )
  if (nzchar(subtitle_text)) {
    grid.text(
      subtitle_text,
      x = 0.5,
      y = unit(0.962, "npc"),
      gp = gpar(fontfamily = "Times", fontsize = subtitle_fontsize, col = "#6B7280")
    )
  }
  if (nzchar(footnote_text)) {
    grid.text(
      footnote_text,
      x = 0.5,
      y = unit(0.018, "npc"),
      gp = gpar(fontfamily = "Times", fontsize = max(subtitle_fontsize - 1, 8.5), col = "#4B5563")
    )
  }
  layout_vp <- viewport(
    x = 0.5,
    y = 0.485,
    width = 0.985,
    height = 0.855,
    layout = grid.layout(n_rows, n_cols)
  )
  pushViewport(layout_vp)
  for (idx in seq_len(n_rows * n_cols)) {
    row_idx <- ((idx - 1) %/% n_cols) + 1
    col_idx <- ((idx - 1) %% n_cols) + 1
    if (!is.null(plot_list[[idx]])) {
      pushViewport(viewport(layout.pos.row = row_idx, layout.pos.col = col_idx))
      grid.draw(ggplotGrob(plot_list[[idx]]))
      upViewport(1)
    }
  }
  upViewport(1)
}

save_singlepage_plot <- function(output_path, plot_list, n_rows, n_cols, title_text, subtitle_text, footnote_text, width, height, pointsize, title_fontsize, subtitle_fontsize) {
  ext <- tolower(tools::file_ext(output_path))
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  if (ext == "pdf") {
    grDevices::pdf(file = output_path, width = width, height = height, family = "Times", pointsize = pointsize, onefile = FALSE)
  } else if (ext == "png") {
    grDevices::png(filename = output_path, width = width, height = height, units = "in", res = 320, type = "cairo")
  } else if (ext == "svg") {
    grDevices::svg(filename = output_path, width = width, height = height, pointsize = pointsize)
  } else {
    stop(sprintf("Unsupported output extension: %s", ext))
  }
  on.exit(grDevices::dev.off(), add = TRUE)
  render_page(plot_list, n_rows, n_cols, title_text, subtitle_text, footnote_text, title_fontsize, subtitle_fontsize)
}

data_path <- payload$data_path
target_column <- payload$target_column
features <- unlist(payload$features)
source_columns <- if (!is.null(payload$source_columns)) unlist(payload$source_columns) else features
display_features <- if (!is.null(payload$display_features)) unlist(payload$display_features) else features
preprocessing_note <- if (!is.null(payload$preprocessing_note)) as.character(payload$preprocessing_note) else ""
output_pdf <- payload$output_pdf
output_png <- payload$output_png
output_svg <- payload$output_svg
summary_path <- payload$summary_path
line_color <- payload$line_color
line_fill <- payload$line_fill
scatter_alpha <- as.numeric(payload$scatter_alpha)
fontsize <- max(as.numeric(payload$fontsize), 10)
title_fontsize <- max(as.numeric(payload$title_fontsize), 17)
subtitle_fontsize <- max(as.numeric(payload$subtitle_fontsize), 11)
label_fontsize <- max(as.numeric(payload$label_fontsize), 12)
font_family <- "Times"

df <- read.csv(data_path, check.names = FALSE, stringsAsFactors = FALSE)
required_columns <- c(source_columns, target_column)
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
layout_dims <- choose_layout(n_feat)
n_cols <- layout_dims[1]
n_rows <- layout_dims[2]

width_in <- max(11.5, 3.9 * n_cols)
height_in <- max(7.0, 3.25 * n_rows + 1.0)

plot_list <- vector("list", n_rows * n_cols)
summary_rows <- vector("list", n_feat)
log_transform_labels <- character(0)

for (i in seq_along(features)) {
  feature <- features[[i]]
  source_column <- source_columns[[i]]
  display_name <- normalize_display_name(display_features[[i]])
  x_raw <- suppressWarnings(as.numeric(df[[source_column]]))
  mask <- is.finite(x_raw) & is.finite(y_pos_full)
  x <- x_raw[mask]
  y <- y_pos_full[mask]

  shapiro_p <- run_shapiro_safely(x)
  skewness_value <- compute_sample_skewness(x)
  log_transform_applied <- should_apply_log_transform(x, shapiro_p, skewness_value)
  if (log_transform_applied) {
    x <- log(x)
    log_transform_labels <- c(log_transform_labels, display_name)
  }

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
    plot_list[[i]] <- plot_failure_panel(display_name, "Unable to place stable RCS knots", fontsize)
    summary_rows[[i]] <- list(
      feature = feature,
      display_feature = display_name,
      status = "failed",
      reason = "Unable to place stable RCS knots"
    )
    next
  }

  boundary_lo <- as.numeric(knot_positions[1])
  internal_knot <- as.numeric(knot_positions[2])
  boundary_hi <- as.numeric(knot_positions[3])
  median_val <- as.numeric(median(x, na.rm = TRUE))
  if (!(boundary_lo < internal_knot && internal_knot < boundary_hi)) {
    plot_list[[i]] <- plot_failure_panel(display_name, "Unable to order boundary/internal knots", fontsize)
    summary_rows[[i]] <- list(
      feature = feature,
      display_feature = display_name,
      status = "failed",
      reason = "Unable to order boundary/internal knots"
    )
    next
  }

  fit_warning_count <- 0L
  fit_warning_msgs <- character(0)
  p_nonlin <- NA_real_
  p_overall <- NA_real_
  x_range <- hi - lo
  p_label_x <- hi - 0.01 * x_range
  p_label_y <- 4.82
  stat_label_size <- (fontsize / 3) * 0.85

  withCallingHandlers(
    {
      df_model <- data.frame(
        outcome = factor(y, levels = c(0, 1), labels = c("0", "1")),
        exposure = x
      )
      dd_name <- sprintf("ddist_rcs_preview_%d", i)
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
        fontsize = fontsize,
        fontfamily = font_family,
        linecolor = line_color,
        alpha = 0.24,
        xbreaks = pretty(c(lo, hi)),
        ybreaks = pretty(c(0, 5)),
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
  turning_point <- find_or1_crossing(plot_data, median_val)
  turning_x <- turning_point$x
  turning_label <- format_axis_value(turning_x)
  turning_hjust <- if (!is.finite(turning_x)) {
    0.5
  } else if (turning_x <= lo + 0.12 * x_range) {
    0
  } else if (turning_x >= hi - 0.12 * x_range) {
    1
  } else {
    0.5
  }
  plot_obj$coordinates <- coord_cartesian(xlim = c(lo, hi), ylim = c(0, 5), clip = "on")
  plot_obj$scales$scales <- Filter(
    function(scale_obj) !any(c("x", "y") %in% scale_obj$aesthetics),
    plot_obj$scales$scales
  )
  plot_obj <- suppressWarnings(
    plot_obj +
      scale_x_continuous(
        breaks = pretty(c(lo, hi)),
        expand = expansion(mult = c(0.02, 0.02))
      ) +
      scale_y_continuous(
        breaks = pretty(c(0, 5)),
        expand = expansion(mult = c(0.00, 0.02))
      ) +
      annotate(
        "text",
        x = p_label_x,
        y = p_label_y,
        label = paste0(
          "Overall p ", format_p_value(p_overall), "\n",
          "Non-linearity p ", format_p_value(p_nonlin)
        ),
        hjust = 1,
        vjust = 1,
        family = font_family,
        size = stat_label_size,
        lineheight = 1.05,
        colour = axis_text_color
      ) +
      {
        if (is.finite(turning_x)) {
          geom_vline(
            xintercept = turning_x,
            linetype = "22",
            linewidth = 0.55,
            color = "#6B7280",
            alpha = 0.95
          )
        }
      } +
      {
        if (is.finite(turning_x)) {
          annotate(
            "segment",
            x = turning_x,
            xend = turning_x,
            y = 0,
            yend = 0.18,
            linewidth = 0.45,
            color = "#6B7280"
          )
        }
      } +
      {
        if (is.finite(turning_x)) {
          annotate(
            "label",
            x = turning_x,
            y = 0.26,
            label = turning_label,
            hjust = turning_hjust,
            vjust = 0.5,
            family = font_family,
            size = fontsize / 3.6,
            color = "#4B5563",
            label.size = 0.15,
            fill = "white",
            alpha = 0.95
          )
        }
      } +
      labs(x = display_name, y = "Odds Ratio (95% CI)") +
      theme_minimal(base_family = font_family, base_size = fontsize) +
      theme(
        axis.title.x = element_text(size = label_fontsize - 0.5, face = "bold", colour = text_color),
        axis.title.y = element_text(size = label_fontsize - 0.5, face = "bold", colour = text_color),
        axis.text = element_text(size = fontsize - 0.3, colour = axis_text_color),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = grid_color, linewidth = 0.28),
        panel.border = element_rect(color = panel_border_color, fill = NA, linewidth = 0.6),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(t = 8, r = 10, b = 6, l = 8)
      )
  )
  plot_list[[i]] <- plot_obj

  summary_rows[[i]] <- list(
    feature = feature,
    source_column = source_column,
    display_feature = display_name,
    status = "completed",
    shapiro_p = shapiro_p,
    skewness = skewness_value,
    log_transform_applied = log_transform_applied,
    p_overall = p_overall,
    p_nonlinearity = p_nonlin,
    n_samples = length(x),
    boundary_knot_p10 = boundary_lo,
    internal_knot_p50 = internal_knot,
    boundary_knot_p90 = boundary_hi,
    reference_value_median = median_val,
    turning_point_x = turning_x,
    turning_point_kind = turning_point$kind,
    x_plot_min_p05 = lo,
    x_plot_max_p95 = hi,
    y_plot_min = if ("lower" %in% names(plot_data)) min(plot_data$lower, na.rm = TRUE) else NA_real_,
    y_plot_max = if ("upper" %in% names(plot_data)) max(plot_data$upper, na.rm = TRUE) else NA_real_,
    fit_warning_count = fit_warning_count,
    fit_warning_messages = fit_warning_msgs
  )
}

title_text <- "Restricted Cubic Spline Risk Curves of Winner-Panel Metabolites"
subtitle_text <- ""
log_transform_labels <- unique(log_transform_labels)
log_transform_note <- if (length(log_transform_labels) > 0) {
  sprintf(
    "Additional plotting log transform applied after Shapiro-Wilk/skewness check: %s.",
    paste(log_transform_labels, collapse = ", ")
  )
} else {
  "Additional plotting log transform: none after Shapiro-Wilk/skewness check."
}
footnote_text <- trimws(paste(preprocessing_note, log_transform_note))

save_singlepage_plot(output_pdf, plot_list, n_rows, n_cols, title_text, subtitle_text, footnote_text, width_in, height_in, fontsize, title_fontsize, subtitle_fontsize)
save_singlepage_plot(output_png, plot_list, n_rows, n_cols, title_text, subtitle_text, footnote_text, width_in, height_in, fontsize, title_fontsize, subtitle_fontsize)
save_singlepage_plot(output_svg, plot_list, n_rows, n_cols, title_text, subtitle_text, footnote_text, width_in, height_in, fontsize, title_fontsize, subtitle_fontsize)

summary_payload <- list(
  positive_class = positive_class,
  prevalence = prevalence,
  n_features = n_feat,
  n_cols = n_cols,
  n_rows = n_rows,
  backend = "r",
  engine = "plotRCS",
  footnote = footnote_text,
  outputs = list(
    pdf = output_pdf,
    png = output_png,
    svg = output_svg
  ),
  features = summary_rows
)
write_json(summary_payload, summary_path, pretty = TRUE, auto_unbox = TRUE, null = "null")

message(sprintf("RCS single-page panels saved to: %s", output_pdf))
