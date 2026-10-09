#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript plot_phase2_incremental_value_preview.R <repeat_result.json> <output_prefix>")
}

repeat_result_path <- args[[1]]
output_prefix <- args[[2]]

payload <- fromJSON(repeat_result_path, simplifyVector = TRUE)
inc <- payload$repeat_summary$internal_oof_metrics_complete$incremental_value
if (is.null(inc) || identical(inc, list()) || !isTRUE(inc$enabled)) {
  inc <- payload$repeat_summary$internal_metrics$incremental_value
}

if (is.null(inc) || identical(inc, list()) || !isTRUE(inc$enabled)) {
  stop("incremental_value is not available in repeat_result.json")
}

compute_continuous_nri <- function(y_true, p_old, p_new, tie_tolerance = 1e-12) {
  delta <- p_new - p_old
  event_mask <- y_true == 1
  nonevent_mask <- y_true == 0

  event_up <- mean(delta[event_mask] > tie_tolerance)
  event_down <- mean(delta[event_mask] < -tie_tolerance)
  nonevent_down <- mean(delta[nonevent_mask] < -tie_tolerance)
  nonevent_up <- mean(delta[nonevent_mask] > tie_tolerance)

  (event_up - event_down) + (nonevent_down - nonevent_up)
}

compute_idi <- function(y_true, p_old, p_new) {
  event_mask <- y_true == 1
  nonevent_mask <- y_true == 0
  old_slope <- mean(p_old[event_mask]) - mean(p_old[nonevent_mask])
  new_slope <- mean(p_new[event_mask]) - mean(p_new[nonevent_mask])
  new_slope - old_slope
}

bootstrap_metric <- function(df, metric_name, n_bootstrap = 200, seed = 42) {
  set.seed(seed)
  event_idx <- which(df$y_true == 1)
  nonevent_idx <- which(df$y_true == 0)

  values <- numeric(n_bootstrap)
  for (i in seq_len(n_bootstrap)) {
    sampled_events <- sample(event_idx, length(event_idx), replace = TRUE)
    sampled_nonevents <- sample(nonevent_idx, length(nonevent_idx), replace = TRUE)
    sampled <- df[c(sampled_events, sampled_nonevents), , drop = FALSE]
    if (metric_name == "Continuous NRI") {
      values[i] <- compute_continuous_nri(sampled$y_true, sampled$p_old, sampled$p_new)
    } else {
      values[i] <- compute_idi(sampled$y_true, sampled$p_old, sampled$p_new)
    }
  }
  values
}

empirical_p_value <- function(values) {
  n <- length(values)
  if (n <= 0) {
    return(NA_real_)
  }
  p_left <- (sum(values <= 0) + 1) / (n + 1)
  p_right <- (sum(values >= 0) + 1) / (n + 1)
  min(1, 2 * min(p_left, p_right))
}

repeat_dir <- dirname(repeat_result_path)
winner_scores_path <- file.path(repeat_dir, "_runtime", "phase1", "legacy", "artifacts", "phase2_winner_scores.json")
winner_payload <- fromJSON(winner_scores_path, simplifyVector = TRUE)

winner_preds <- winner_payload$cv_predictions$predictions
baseline_preds <- winner_payload$prior_anchor_union_evaluation$phase1_model_evaluation$phase1_panel_baseline_evaluation$cv_predictions$predictions

winner_df <- data.frame(
  sample_id = winner_preds$sample_id,
  fold = winner_preds$fold,
  y_true = winner_preds$true_label_encoded,
  p_new = winner_preds$pred_proba_class1
)
baseline_df <- data.frame(
  sample_id = baseline_preds$sample_id,
  fold = baseline_preds$fold,
  y_true = baseline_preds$true_label_encoded,
  p_old = baseline_preds$pred_proba_class1
)

merged <- merge(
  baseline_df,
  winner_df[, c("sample_id", "fold", "p_new")],
  by = c("sample_id", "fold"),
  all = FALSE,
  sort = TRUE
)

nri_boot <- bootstrap_metric(merged, "Continuous NRI", n_bootstrap = as.integer(inc$nri$n_bootstrap), seed = 42)
idi_boot <- bootstrap_metric(merged, "IDI", n_bootstrap = as.integer(inc$idi$n_bootstrap), seed = 43)

nri_value <- compute_continuous_nri(merged$y_true, merged$p_old, merged$p_new)
idi_value <- compute_idi(merged$y_true, merged$p_old, merged$p_new)
nri_ci <- as.numeric(stats::quantile(nri_boot, c(0.025, 0.975)))
idi_ci <- as.numeric(stats::quantile(idi_boot, c(0.025, 0.975)))
nri_p_exact <- empirical_p_value(nri_boot)
idi_p_exact <- empirical_p_value(idi_boot)

plot_df <- data.frame(
  metric = factor(c("Continuous NRI", "IDI"), levels = c("Continuous NRI", "IDI")),
  estimate = c(nri_value, idi_value),
  ci_low = c(nri_ci[1], idi_ci[1]),
  ci_high = c(nri_ci[2], idi_ci[2]),
  fill = c("#C97C5D", "#6F8FB1"),
  label = c(
    sprintf(
      "%.3f  [%.3f, %.3f]  p = %.3f",
      nri_value, nri_ci[1], nri_ci[2], nri_p_exact
    ),
    sprintf(
      "%.3f  [%.3f, %.3f]  p = %.3f",
      idi_value, idi_ci[1], idi_ci[2], idi_p_exact
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
  nri_value - (mean((merged$p_new - merged$p_old)[merged$y_true == 0] < -1e-12) - mean((merged$p_new - merged$p_old)[merged$y_true == 0] > 1e-12)),
  mean((merged$p_new - merged$p_old)[merged$y_true == 0] < -1e-12) - mean((merged$p_new - merged$p_old)[merged$y_true == 0] > 1e-12),
  mean((merged$p_new - merged$p_old)[merged$y_true == 1]),
  mean((merged$p_new - merged$p_old)[merged$y_true == 0])
)

p <- ggplot(plot_df, aes(y = metric, x = estimate)) +
  geom_vline(xintercept = left_breaks[left_breaks != 0], color = "#E5E7EB", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "#94A3B8", linetype = "solid", linewidth = 0.95) +
  geom_segment(
    aes(x = ci_low, xend = ci_high, y = metric, yend = metric),
    linewidth = 1.15,
    color = "#6B7280",
    lineend = "round"
  ) +
  geom_segment(
    aes(x = ci_low, xend = ci_low, y = as.numeric(metric) - 0.10, yend = as.numeric(metric) + 0.10),
    linewidth = 0.75,
    color = "#6B7280"
  ) +
  geom_segment(
    aes(x = ci_high, xend = ci_high, y = as.numeric(metric) - 0.10, yend = as.numeric(metric) + 0.10),
    linewidth = 0.75,
    color = "#6B7280"
  ) +
  geom_point(
    aes(fill = fill),
    shape = 21,
    size = 4.8,
    stroke = 0.6,
    color = "#1F2937",
    show.legend = FALSE
  ) +
  annotate(
    "rect",
    xmin = label_panel_xmin,
    xmax = label_panel_xmax,
    ymin = 0.45,
    ymax = 2.55,
    fill = "white",
    color = NA
  ) +
  geom_text(
    aes(x = label_text_x, label = label),
    hjust = 0,
    size = 4.0,
    family = "sans",
    color = "#111827"
  ) +
  annotate(
    "text",
    x = mean(c(x_min, x_max)),
    y = 2.19,
    hjust = 0.5,
    size = 3.35,
    family = "sans",
    color = "#4B5563",
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
    title = "Incremental Value of the Phase 2 Panel",
    x = "Improvement over Phase 1",
    y = NULL
  ) +
  theme_minimal(base_family = "sans", base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.y = element_text(size = 12, color = "#111827", face = "bold"),
    axis.text.x = element_text(size = 10.5, color = "#374151"),
    axis.title.x = element_text(size = 11.2, color = "#111827", face = "bold"),
    plot.title = element_text(size = 15, face = "bold", hjust = 0.5, color = "#111827"),
    plot.subtitle = element_blank(),
    plot.caption = element_blank(),
    plot.margin = margin(16, 116, 16, 24)
  )

png_path <- paste0(output_prefix, ".png")
pdf_path <- paste0(output_prefix, ".pdf")

ggsave(png_path, plot = p, width = 12.4, height = 5.6, dpi = 320, bg = "white")
ggsave(pdf_path, plot = p, width = 12.4, height = 5.6, bg = "white")

cat(png_path, "\n")
cat(pdf_path, "\n")
