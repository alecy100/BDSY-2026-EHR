# ============================================================
# SHARED HELPER: Rubin's rules pooling across the 5 imputations
#
# Works for any model with a named-coefficient-vector + named-variance-
# vector representation -- so the LMM's fixed effects (fixef()/vcov())
# and the logistic regression's coefficients (coef()/vcov()) can both be
# pooled with the same underlying math, rather than needing separate
# pooling code (or an extra dependency like broom.mixed) for each.
#
# Rubin's rules (m imputations, term-by-term):
#   qbar      = mean of the m point estimates                  (pooled estimate)
#   ubar      = mean of the m within-imputation variances       (usual sampling error)
#   b         = variance OF the m point estimates ACROSS imputations
#   total_var = ubar + (1 + 1/m) * b
#   df        = Barnard-Rubin adjusted degrees of freedom
# ============================================================

pool_rubin <- function(estimate_mat, var_mat) {
  # estimate_mat, var_mat: m x p matrices (rows = imputations, cols = terms).
  # var_mat holds the SQUARED standard error (i.e. variance) of each estimate.
  m      <- nrow(estimate_mat)
  qbar   <- colMeans(estimate_mat)
  ubar   <- colMeans(var_mat)
  b      <- apply(estimate_mat, 2, var)
  total_var <- ubar + (1 + 1 / m) * b
  se     <- sqrt(total_var)

  r      <- (1 + 1 / m) * b / ubar
  df_old <- (m - 1) * (1 + 1 / r)^2
  df_old[!is.finite(df_old)] <- (m - 1)   # guard against ubar == 0 (e.g. a constant term)

  t_stat <- qbar / se
  p_val  <- 2 * pt(-abs(t_stat), df = df_old)

  data.frame(
    term        = colnames(estimate_mat),
    estimate    = qbar,
    std.error   = se,
    within_var  = ubar,
    between_var = b,
    df          = df_old,
    statistic   = t_stat,
    p.value     = p_val,
    row.names   = NULL
  )
}

# convenience wrapper for a list of glm objects (the mortality logistic models)
pool_glm_list <- function(fit_list) {
  terms_common <- Reduce(intersect, lapply(fit_list, function(f) names(coef(f))))
  est_mat <- t(sapply(fit_list, function(f) coef(f)[terms_common]))
  var_mat <- t(sapply(fit_list, function(f) diag(vcov(f))[terms_common]))
  colnames(est_mat) <- colnames(var_mat) <- terms_common
  pool_rubin(est_mat, var_mat)
}

# convenience wrapper for a list of lmer fits (pools the FIXED effects only --
# random-effect variance components aren't pooled the same way and are
# reported per-imputation instead, since they aren't testable point estimates)
pool_lmer_fixef_list <- function(fit_list) {
  terms_common <- Reduce(intersect, lapply(fit_list, function(f) names(lme4::fixef(f))))
  est_mat <- t(sapply(fit_list, function(f) lme4::fixef(f)[terms_common]))
  var_mat <- t(sapply(fit_list, function(f) diag(as.matrix(vcov(f)))[terms_common]))
  colnames(est_mat) <- colnames(var_mat) <- terms_common
  pool_rubin(est_mat, var_mat)
}
