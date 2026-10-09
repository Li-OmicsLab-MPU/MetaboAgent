get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), winslash = "/", mustWork = FALSE)
}

FONT_FAMILY_ALIASES <- c(
  "times" = "serif",
  "times roman" = "serif",
  "times new roman" = "serif",
  "times new roman ps" = "serif"
)

normalize_font_family <- function(font_family) {
  requested <- trimws(as.character(font_family %||% ""))
  if (!nzchar(requested)) {
    return(requested)
  }
  key <- tolower(requested)
  alias_value <- FONT_FAMILY_ALIASES[key]
  if (!is.na(alias_value)) {
    return(unname(alias_value))
  }
  requested
}

registered_font_families <- function() {
  families <- c(names(grDevices::pdfFonts()), names(grDevices::postscriptFonts()))
  if (exists("X11Fonts", where = asNamespace("grDevices"), inherits = FALSE)) {
    families <- c(families, names(grDevices::X11Fonts()))
  }
  unique(families[nzchar(families)])
}

is_times_family_request <- function(font_family) {
  normalized <- tolower(trimws(as.character(font_family %||% "")))
  normalized <- gsub("[-_]+", " ", normalized)
  normalized <- gsub("\\s+", " ", normalized)
  normalized %in% c("times", "times roman", "times new roman")
}

resolve_font_family <- function(font_family = NULL, default_family = "sans") {
  requested <- normalize_font_family(font_family)
  generic_families <- c("sans", "serif", "mono")
  fallback <- tolower(trimws(as.character(default_family %||% "sans")))
  if (!fallback %in% generic_families) {
    fallback <- "sans"
  }
  if (!nzchar(requested)) {
    return(fallback)
  }

  normalized <- tolower(trimws(requested))
  if (normalized %in% generic_families) {
    return(normalized)
  }

  # Ignore device-specific font names and fall back to a generic family
  # so PDF/PNG/SVG export can succeed on machines without that font.
  fallback
}

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs) || length(lhs) == 0 || (is.atomic(lhs) && all(is.na(lhs)))) {
    rhs
  } else {
    lhs
  }
}
