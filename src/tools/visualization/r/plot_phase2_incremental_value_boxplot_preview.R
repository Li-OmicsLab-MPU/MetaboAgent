#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript plot_phase2_incremental_value_boxplot_preview.R <phase2_winner_scores.json> <output_prefix>")
}

winner_scores_path <- args[[1]]
output_prefix <- args[[2]]

payload <- fromJSON(winner_scores_path, simplifyVector = TRUE)

winner_preds <- payload$cv_predictions$predictions
baseline_preds <- payload$prior_anchor_union_evaluation$phase1_model_evaluation$cv_predictions$predictions
inc <- payload$incremental_value

if (is.null(winner_preds) || is.null(baseline_preds) || is.null(inc) || !isTRUE(inc$enabled)) {
  stop("phase2_winner_scores.json does not contain baseline/winner predictions plus incremental_value.")
}

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

if (nrow(merged) == 0) {
  stop("No overlapping prediction rows found between baseline and winner payloads.")
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
    sampled_idx <- c(sampled_events, sampled_nonevents)
    sampled <- df[sampled_idx, , drop = FALSE]

    if (metric_name == "Continuous NRI") {
      values[i] <- compute_continuous_nri(sampled$y_true, sampled$p_old, sampled$p_new)
    } else {
      values[i] <- compute_idi(sampled$y_true, sampled$p_old, sampled$p_new)
    }
  }
  values
}

nri_boot <- bootstrap_metric(merged, "Continuous NRI", n_bootstrap = 200, seed = 42)
idi_boot <- bootstrap_metric(merged, "IDI", n_bootstrap = 200, seed = 43)

obs_df <- data.frame(
  metric = factor(c("Continuous NRI", "IDI"), levels = c("Continuous NRI", "IDI")),
  estimate = c(as.numeric(inc$nri$value), as.numeric(inc$idi$value)),
  ci_low = c(as.numeric(inc$nri$ci_95[1]), as.numeric(inc$idi$ci_95[1])),
  ci_high = c(as.numeric(inc$nri$ci_95[2]), as.numeric(inc$idi$ci_95[2])),
  stringsAsFactors = FALSE
)

boot_df <- rbind(
  data.frame(metric = "Continuous NRI", value = nri_boot),
  data.frame(metric = "IDI", value = idi_boot)
)
boot_df$metric <- factor(boot_df$metric, levels = c("Continuous NRI", "IDI"))

header_label <- sprintf(
  "repeat_001 OOF bootstrap = 200   |   n = %d   |   Phase 1: %d biomarkers -> Phase 2: %d biomarkers",
  as.integer(inc$n_samples),
  as.integer(inc$baseline_feature_count),
  as.integer(inc$new_feature_count)
)

subtitle_label <- sprintf(
  "Boxplots show stratified bootstrap replicates; diamonds denote observed estimates; adjacent text reports 95%% CI."
)

label_df <- data.frame(
  metric = factor(c("Continuous NRI", "IDI"), levels = c("Continuous NRI", "IDI")),
  x = c(max(nri_boot, na.rm = TRUE), max(idi_boot, na.rm = TRUE)),
  label = c(
    sprintf("obs=%.3f\n95%% CI [%.3f, %.3f]", obs_df$estimate[1], obs_df$ci_low[1], obs_df$ci_high[1]),
    sprintf("obs=%.3f\n95%% CI [%.3f, %.3f]", obs_df$estimate[2], obs_df$ci_low[2], obs_df$ci_high[2])
  )
)

x_min <- min(c(boot_df$value, obs_df$ci_low, 0), na.rm = TRUE)
x_max <- max(c(boot_df$value, obs_df$ci_high, 0), na.rm = TRUE)
x_span <- x_max - x_min
if (!is.finite(x_span) || x_span <= 0) {
  x_span <- 0.1
}
label_df$x <- label_df$x + 0.08 * x_span

p <- ggplot(boot_df, aes(x = value, y = metric, fill = metric)) +
  geom_vline(xintercept = 0, color = "#9CA3AF", linetype = "22", linewidth = 0.6) +
  geom_boxplot(
    width = 0.38,
    outlier.shape = NA,
    alpha = 0.92,
    color = "#374151",
    linewidth = 0.55
  ) +
  geom_jitter(
    width = 0,
    height = 0.10,
    size = 1.25,
    alpha = 0.24,
    color = "#6B7280"
  ) +
  geom_point(
    data = obs_df,
    aes(x = estimate, y = metric),
    inherit.aes = FALSE,
    shape = 23,
    size = 4.2,
    stroke = 0.6,
    fill = "#C97C5D",
    color = "#111827"
  ) +
  geom_text(
    data = label_df,
    aes(x = x, y = metric, label = label),
    inherit.aes = FALSE,
    hjust = 0,
    size = 3.85,
    family = "sans",
    color = "#111827",
    lineheight = 0.95
  ) +
  scale_fill_manual(values = c("Continuous NRI" = "#D29A7A", "IDI" = "#8AA5C2")) +
  scale_x_continuous(
    limits = c(x_min - 0.05 * x_span, x_max + 0.42 * x_span),
    expand = c(0, 0),
    labels = function(x) sprintf("%.2f", x)
  ) +
  labs(
    title = "Bootstrap Distribution of Incremental Value",
    subtitle = "Phase 2 biomarker panel relative to the Phase 1 baseline panel",
    x = "Bootstrap Estimate",
    y = NULL,
    caption = "Continuous NRI and IDI were recomputed from aligned out-of-fold probabilities using 200 stratified bootstrap resamples."
  ) +
  annotate(
    "label",
    x = x_min - 0.03 * x_span,
    y = 2.42,
    hjust = 0,
    vjust = 1,
    linewidth = 0,
    size = 4.0,
    family = "sans",
    color = "#111827",
    fill = "#F8F5F2",
    label.padding = unit(c(0.20, 0.32, 0.20, 0.32), "lines"),
    label = header_label
  ) +
  annotate(
    "text",
    x = x_min - 0.03 * x_span,
    y = 2.18,
    hjust = 0,
    size = 3.55,
    family = "sans",
    color = "#4B5563",
    label = subtitle_label
  ) +
  theme_minimal(base_family = "sans", base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = 0.5),
    axis.text.y = element_text(size = 12, color = "#111827", face = "bold"),
    axis.text.x = element_text(size = 10.5, color = "#111827"),
    axis.title.x = element_text(size = 11.5, color = "#111827", face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5, color = "#111827"),
    plot.subtitle = element_text(size = 11.5, hjust = 0.5, color = "#374151"),
    plot.caption = element_text(size = 9.5, color = "#6B7280"),
    legend.position = "none",
    plot.margin = margin(18, 92, 18, 24)
  )

png_path <- paste0(output_prefix, ".png")
pdf_path <- paste0(output_prefix, ".pdf")

ggsave(png_path, plot = p, width = 12.8, height = 5.8, dpi = 320, bg = "white")
ggsave(pdf_path, plot = p, width = 12.8, height = 5.8, bg = "white")

cat(png_path, "\n")
cat(pdf_path, "\n")
