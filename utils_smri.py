# ============================================================
# REMANO sMRI utilities
# General stats + ROI + ratios + deltas + prediction + VBM
# ============================================================

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from nilearn.image import load_img, math_img, mean_img, resample_to_img
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.masking import compute_brain_mask, apply_mask
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table


# ============================================================
# 1) BASIC HELPERS
# ============================================================

def parse_sub_ses(filename: str):
    """
    Parse subject and session from filenames like:
    s8mwp1rsub-94_ses-2_T1w.nii.gz
    """
    m = re.search(r"sub-(\d+).*ses-([A-Za-z0-9]+)", filename)
    if not m:
        raise ValueError(f"Could not parse subject/session from: {filename}")

    subject = int(m.group(1))
    ses_raw = m.group(2)

    try:
        session = int(ses_raw)
    except ValueError:
        session = ses_raw

    return subject, session


def ensure_categorical(df: pd.DataFrame, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def safe_dropna(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.copy()
    return df.dropna(subset=cols).copy()


# ============================================================
# 3) GENERIC MODEL FITTER
# ============================================================

def fit_model(
    data: pd.DataFrame,
    formula: str,
    model_type="ols",
    group_col="subject",
    reml=False,
    method="lbfgs",
    dropna_cols=None,
    verbose=False,
    **fit_kwargs,
):
    """
    Generic wrapper for OLS and MixedLM.
    """
    df = data.copy()

    needed = []
    if dropna_cols is not None:
        needed.extend(dropna_cols)
    if model_type == "mixedlm":
        needed.append(group_col)

    df = safe_dropna(df, list(set(needed)))
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    df = df.reset_index(drop=True)

    if model_type == "ols":
        model = smf.ols(formula, data=df).fit()

    elif model_type == "mixedlm":
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df[group_col]
        ).fit(
            reml=reml,
            method=method,
            maxiter=200,
            disp=False,
            **fit_kwargs
        )

    else:
        raise ValueError("model_type must be 'ols' or 'mixedlm'")

    if verbose:
        print(model.summary())

    return model

def summarize_model(model, keep_terms=None, outcome_name=None):
    """
    Convert fitted statsmodels object into one flat summary row.
    """
    row = {
        "outcome": outcome_name,
        "n": len(model.model.data.frame),
    }

    if hasattr(model, "rsquared"):
        row["R2"] = getattr(model, "rsquared", np.nan)
        row["adj_R2"] = getattr(model, "rsquared_adj", np.nan)

    if hasattr(model, "fvalue"):
        row["F"] = getattr(model, "fvalue", np.nan)
        row["p_model"] = getattr(model, "f_pvalue", np.nan)

    params = model.params
    pvals = model.pvalues

    if keep_terms is None:
        terms = params.index
    else:
        terms = keep_terms

    for term in terms:
        row[f"beta_{term}"] = params.get(term, np.nan)
        row[f"p_{term}"] = pvals.get(term, np.nan)

    return row


def tidy_roi_results(df, predictor):
    """
    Convert run_roi_models / run_roi_screen output
    into a tidy dataframe with columns:
    roi | var | beta | p | p_fdr | n
    """
    beta_col = f"beta_{predictor}"
    p_col = f"p_{predictor}"
    fdr_col = f"{p_col}_fdr"

    out = pd.DataFrame({
        "roi": df["roi"],
        "var": predictor,
        "beta": df.get(beta_col),
        "p": df.get(p_col),
        "p_fdr": df.get(fdr_col),
        "n": df.get("n"),
    })
    return out


def tidy_interaction_results(df, predictor):
    """
    Convert interaction output into tidy format.
    """
    out = pd.DataFrame({
        "roi": df["roi"],
        "var": predictor,
        "beta": df.get("beta_session"),
        "p": df.get("p_interaction"),
        "p_fdr": df.get("p_interaction_fdr"),
        "n": df.get("n_subjects"),
    })
    return out


# ============================================================
# 4) GLOBAL MODELS
# ============================================================

def run_global_models(
    data: pd.DataFrame,
    outcomes,
    formula_template: str,
    model_type="ols",
    group_col="subject",
    dropna_extra=None,
    terms_of_interest=None,
    reml=False,
    method="lbfgs",
):
    """
    Run the same model across multiple global outcomes.
    Example:
        outcomes = ["GM_total_ml", "WM_total_ml", "CSF_total_ml"]
        formula_template = "{Y} ~ C(group) + age + TIV_ml"
    """
    if dropna_extra is None:
        dropna_extra = []

    rows = []

    for outcome in outcomes:
        formula = formula_template.format(Y=outcome)

        try:
            model = fit_model(
                data=data,
                formula=formula,
                model_type=model_type,
                group_col=group_col,
                reml=reml,
                method=method,
                dropna_cols=[outcome] + list(dropna_extra),
            )

            row = summarize_model(
                model,
                keep_terms=terms_of_interest,
                outcome_name=outcome,
            )
            row["formula"] = formula
            rows.append(row)

        except Exception as e:
            # fallback for MixedLM singular matrix
            if model_type == "mixedlm":
                try:
                    df_tmp = data.copy()
                    formula = formula_template.format(Y=outcome)

                    needed_cols = [outcome, group_col] + list(dropna_extra)
                    df_tmp = df_tmp.replace([np.inf, -np.inf], np.nan)
                    df_tmp = df_tmp.dropna(subset=[c for c in needed_cols if c in df_tmp.columns])

                    model = smf.ols(formula, data=df_tmp).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": df_tmp[group_col]}
                    )

                    row = summarize_model(
                        model,
                        keep_terms=terms_of_interest,
                        outcome_name=outcome,
                    )
                    row["formula"] = formula
                    row["model_used"] = "ols_cluster_subject"
                    row["mixedlm_error"] = str(e)
                    rows.append(row)

                except Exception as e2:
                    rows.append({
                        "outcome": outcome,
                        "formula": formula,
                        "error": f"MixedLM failed: {e}; OLS cluster failed: {e2}",
                    })

            else:
                rows.append({
                    "outcome": outcome,
                    "formula": formula,
                    "error": str(e),
                })

    return pd.DataFrame(rows)



def add_family_fdr(df):
    df = df.copy()
    df["p_fdr"] = np.nan
    for _, idx in df.groupby("family", dropna=False).groups.items():
        valid_idx = df.loc[idx].index[df.loc[idx, "p"].notna()]
        if len(valid_idx) > 0:
            df.loc[valid_idx, "p_fdr"] = multipletests(df.loc[valid_idx, "p"], method="fdr_bh")[1]
    return df

def run_clinical_global_cross_long(df_all: pd.DataFrame, df_long: pd.DataFrame, clinical_vars, outcomes, clinical_families, group_value="AN", baseline_session=1, cross_covariates=("age", "TIV_ml"), long_covariates=("age", "TIV_ml"), add_bmi=False, group_col="subject"):
    if isinstance(clinical_vars, str):
        clinical_vars = [clinical_vars]

    family_lookup = {var: family for family, variables in clinical_families.items() for var in variables}
    cross_covars = list(cross_covariates)
    long_covars = list(long_covariates)

    if add_bmi:
        if "BMI" not in cross_covars:
            cross_covars.append("BMI")
        if "BMI" not in long_covars:
            long_covars.append("BMI")

    df_cross = df_all[(df_all["group"] == group_value) & (df_all["session"] == baseline_session)].copy()
    df_long_sub = df_long[df_long["group"] == group_value].copy()

    cross_results = []
    long_results = []

    for clinical_var in clinical_vars:
        formula_cross = "{Y} ~ " + " + ".join([clinical_var] + cross_covars)

        res_cross = run_global_models(data=df_cross, outcomes=outcomes, formula_template=formula_cross, model_type="ols", group_col=group_col, dropna_extra=[clinical_var] + cross_covars, terms_of_interest=[clinical_var]).copy()

        beta_col = f"beta_{clinical_var}"
        p_col = f"p_{clinical_var}"

        cross_tidy = pd.DataFrame({
            "outcome": res_cross["outcome"],
            "var": clinical_var,
            "family": family_lookup.get(clinical_var, "Other"),
            "beta": res_cross[beta_col] if beta_col in res_cross.columns else np.nan,
            "p": res_cross[p_col] if p_col in res_cross.columns else np.nan,
            "n": res_cross["n"] if "n" in res_cross.columns else np.nan,
            "formula": res_cross["formula"] if "formula" in res_cross.columns else np.nan,
            "analysis": "cross",
        })

        if "error" in res_cross.columns:
            cross_tidy["error"] = res_cross["error"]

        cross_results.append(cross_tidy)

        formula_long = "{Y} ~ C(session) * " + clinical_var + " + " + " + ".join(long_covars)
        interaction_term = f"C(session)[T.2]:{clinical_var}"

        res_long = run_global_models(data=df_long_sub, outcomes=outcomes, formula_template=formula_long, model_type="mixedlm", group_col=group_col, dropna_extra=[clinical_var] + long_covars, terms_of_interest=[interaction_term]).copy()

        beta_col = f"beta_{interaction_term}"
        p_col = f"p_{interaction_term}"

        long_tidy = pd.DataFrame({
            "outcome": res_long["outcome"],
            "var": clinical_var,
            "family": family_lookup.get(clinical_var, "Other"),
            "beta": res_long[beta_col] if beta_col in res_long.columns else np.nan,
            "p": res_long[p_col] if p_col in res_long.columns else np.nan,
            "n": res_long["n"] if "n" in res_long.columns else np.nan,
            "formula": res_long["formula"] if "formula" in res_long.columns else np.nan,
            "analysis": "long",
        })

        if "error" in res_long.columns:
            long_tidy["error"] = res_long["error"]

        long_results.append(long_tidy)

    cross_df = pd.concat(cross_results, ignore_index=True) if cross_results else pd.DataFrame()
    long_df = pd.concat(long_results, ignore_index=True) if long_results else pd.DataFrame()

    if not cross_df.empty:
        cross_df = add_family_fdr(cross_df)

    if not long_df.empty:
        long_df = add_family_fdr(long_df)

    return cross_df, long_df


# ============================================================
# 5) ROI MODELS
# ============================================================

def run_roi_models(
    data: pd.DataFrame,
    roi_col: str,
    formula: str,
    model_type="ols",
    roi_list=None,
    group_col="subject",
    min_n=5,
    min_subjects=5,
    terms_of_interest=None,
    reml=False,
    method="lbfgs",
    fdr=True,
):
    """
    Generic ROI-wise loop for OLS or MixedLM.
    """
    df = data.copy()

    if roi_list is None:
        roi_list = sorted(df[roi_col].dropna().unique())

    rows = []

    for roi in roi_list:
        df_r = df[df[roi_col] == roi].copy()

        if len(df_r) < min_n:
            continue

        if model_type == "mixedlm":
            if group_col not in df_r.columns or df_r[group_col].nunique() < min_subjects:
                continue

        try:
            model = fit_model(
                data=df_r,
                formula=formula,
                model_type=model_type,
                group_col=group_col,
                reml=reml,
                method=method,
            )

            row = summarize_model(
                model,
                keep_terms=terms_of_interest,
                outcome_name=roi,
            )
            row["roi"] = roi
            row["n_subjects"] = (
                df_r[group_col].nunique() if group_col in df_r.columns else np.nan
            )
            rows.append(row)

        except Exception as e:
            rows.append({
                "roi": roi,
                "error": str(e),
            })

    out = pd.DataFrame(rows)

    if fdr and terms_of_interest is not None and not out.empty:
        for term in terms_of_interest:
            p_col = f"p_{term}"
            if p_col in out.columns:
                mask = out[p_col].notna()
                out[f"{p_col}_fdr"] = np.nan
                if mask.sum() > 0:
                    out.loc[mask, f"{p_col}_fdr"] = multipletests(
                        out.loc[mask, p_col],
                        method="fdr_bh"
                    )[1]

    return out


def run_roi_screen(
    data: pd.DataFrame,
    predictor: str,
    outcome="GM_roi_ml",
    covariates=("age", "TIV_ml"),
    roi_col="roi",
    roi_list=None,
    model_type="ols",
    group_col="subject",
    reml=False,
    method="lbfgs",
):
    """
    Convenient wrapper for repeated clinical ROI screens.
    """
    rhs = [predictor] + list(covariates)
    formula = f"{outcome} ~ " + " + ".join(rhs)

    return run_roi_models(
        data=data,
        roi_col=roi_col,
        formula=formula,
        model_type=model_type,
        roi_list=roi_list,
        group_col=group_col,
        terms_of_interest=[predictor],
        reml=reml,
        method=method,
        fdr=True,
    )


def extract_min_interaction_p(model, prefix="C(session)[T.2]:"):
    """
    Useful for remission/session or subtype/session interactions where
    statsmodels may encode interaction term names with different category labels.
    """
    pvals = model.pvalues
    keys = [k for k in pvals.index if k.startswith(prefix)]
    if len(keys) == 0:
        return np.nan
    return np.min([pvals[k] for k in keys])


def run_roi_interaction_models(
    data: pd.DataFrame,
    roi_col: str,
    formula: str,
    session_term="C(session)[T.2]",
    interaction_prefix="C(session)[T.2]:",
    group_col="subject",
    roi_list=None,
    min_subjects=5,
    reml=False,
    method="lbfgs",
):
    """
    ROI loop specifically for longitudinal interaction models.
    """
    df = data.copy()

    if roi_list is None:
        roi_list = sorted(df[roi_col].dropna().unique())

    rows = []

    for roi in roi_list:
        df_r = df[df[roi_col] == roi].copy()

        if df_r[group_col].nunique() < min_subjects:
            continue

        try:
            model = fit_model(
                data=df_r,
                formula=formula,
                model_type="mixedlm",
                group_col=group_col,
                reml=reml,
                method=method,
                disp=False,
            )

            params = model.params
            pvals = model.pvalues

            rows.append({
                "roi": roi,
                "n_subjects": int(df_r[group_col].nunique()),
                "beta_session": params.get(session_term, np.nan),
                "p_session": pvals.get(session_term, np.nan),
                "p_interaction": extract_min_interaction_p(
                    model,
                    prefix=interaction_prefix
                ),
            })

        except Exception as e:
            rows.append({
                "roi": roi,
                "error": str(e),
            })

    out = pd.DataFrame(rows)

    for p_col in ["p_session", "p_interaction"]:
        if p_col in out.columns:
            mask = out[p_col].notna()
            out[f"{p_col}_fdr"] = np.nan
            if mask.sum() > 0:
                out.loc[mask, f"{p_col}_fdr"] = multipletests(
                    out.loc[mask, p_col],
                    method="fdr_bh"
                )[1]

    return out


clinical_families = {
    "HADS": ["hads_anxiete", "hads_depression", "hads"],
    "EDI2": ["edi2_minceur", "edi2_boulimie", "edi2_insatisfaction", "edi2_inefficacite", "edi2_perfectionnisme", "edi2_interoceptive", "edi2_mefiance", "edi2_maturite", "edi2_ascetisme", "edi2_impulsions", "edi2_insecurite"],
    "YBS": ["ybs_obs", "ybs_comp"],
    "RAADS": ["raads_score", "raads_hypersens", "raads_anxsoc", "raads_mental"],
    "CTQ": ["ctq_score", "ctq_emotionnel", "ctq_physique", "ctq_sexuel", "ctq_negemo", "ctq_negphy"],
    "Cognition": ["brixton_score", "tmt_b_moins_a_score"],
    "DDT": ["ddt_p1_k", "ddt_p1_m", "ddt_p2_k", "ddt_p2_m"],
}

def add_family_fdr(df):
    df = df.copy()
    df["p_fdr"] = np.nan
    for _, idx in df.groupby("family", dropna=False).groups.items():
        valid_idx = df.loc[idx].index[df.loc[idx, "p"].notna()]
        if len(valid_idx) > 0:
            df.loc[valid_idx, "p_fdr"] = multipletests(df.loc[valid_idx, "p"], method="fdr_bh")[1]
    return df

def run_clinical_roi_cross_long(df_all: pd.DataFrame, df_long: pd.DataFrame, clinical_vars, roi_list, clinical_families=clinical_families, outcome="Vgm", roi_col="roi", group_col="subject", group_value="AN", baseline_session=1, cross_covariates=("age", "TIV_ml"), long_covariates=("age", "TIV_ml"), add_bmi=False, session_term="C(session)[T.2]", reml=False, method="lbfgs"):
    if isinstance(clinical_vars, str):
        clinical_vars = [clinical_vars]
    family_lookup = {var: family for family, variables in clinical_families.items() for var in variables}
    cross_results = []
    long_results = []
    cross_covars = list(cross_covariates)
    long_covars = list(long_covariates)
    if add_bmi:
        if "BMI" not in cross_covars:
            cross_covars.append("BMI")
        if "BMI" not in long_covars:
            long_covars.append("BMI")
    df_cross = df_all[(df_all["group"] == group_value) & (df_all["session"] == baseline_session)].copy()
    df_long_sub = df_long[df_long["group"] == group_value].copy()
    for clinical_var in clinical_vars:
        res_cross = run_roi_screen(data=df_cross, predictor=clinical_var, outcome=outcome, covariates=tuple(cross_covars), roi_col=roi_col, roi_list=roi_list, model_type="ols", group_col=group_col, reml=reml, method=method).copy()
        formula_cross = f"{outcome} ~ " + " + ".join([clinical_var] + cross_covars)
        res_cross = tidy_roi_results(res_cross, clinical_var)
        res_cross["var"] = clinical_var
        res_cross["family"] = family_lookup.get(clinical_var, "Other")
        res_cross["analysis"] = "cross"
        res_cross["group"] = group_value
        res_cross["session"] = baseline_session
        res_cross["formula"] = formula_cross
        cross_results.append(res_cross)
        formula_long = f"{outcome} ~ C(session) * {clinical_var} + " + " + ".join(long_covars)
        res_long = run_roi_interaction_models(data=df_long_sub, roi_col=roi_col, formula=formula_long, session_term=session_term, interaction_prefix=f"{session_term}:", group_col=group_col, roi_list=roi_list, reml=reml, method=method).copy()
        res_long = tidy_interaction_results(res_long, clinical_var)
        res_long["var"] = clinical_var
        res_long["family"] = family_lookup.get(clinical_var, "Other")
        res_long["analysis"] = "long"
        res_long["group"] = group_value
        res_long["formula"] = formula_long
        long_results.append(res_long)
    cross_df = pd.concat(cross_results, ignore_index=True) if cross_results else pd.DataFrame()
    long_df = pd.concat(long_results, ignore_index=True) if long_results else pd.DataFrame()
    if not cross_df.empty:
        cross_df["p_fdr"] = np.nan
        for _, idx in cross_df.groupby("family", dropna=False).groups.items():
            valid_idx = cross_df.loc[idx].index[cross_df.loc[idx, "p"].notna()]
            if len(valid_idx) > 0:
                cross_df.loc[valid_idx, "p_fdr"] = multipletests(cross_df.loc[valid_idx, "p"], method="fdr_bh")[1]
    if not long_df.empty:
        long_df["p_fdr"] = np.nan
        for _, idx in long_df.groupby("family", dropna=False).groups.items():
            valid_idx = long_df.loc[idx].index[long_df.loc[idx, "p"].notna()]
            if len(valid_idx) > 0:
                long_df.loc[valid_idx, "p_fdr"] = multipletests(long_df.loc[valid_idx, "p"], method="fdr_bh")[1]
    return cross_df, long_df


# ============================================================
# 6) DELTA / CHANGE DATAFRAMES
# ============================================================

def make_delta_df(
    data: pd.DataFrame,
    value_cols,
    subject_col="subject",
    session_col="session",
    baseline=1,
    followup=2,
):
    """
    Wide delta dataframe for global variables.
    """
    needed = [subject_col, session_col] + list(value_cols)
    df = data[needed].copy()

    wide = df.pivot_table(
        index=subject_col,
        columns=session_col,
        values=value_cols,
        aggfunc="mean"
    )

    out = pd.DataFrame(index=wide.index)

    for col in value_cols:
        if col not in wide.columns.get_level_values(0):
            continue
        if baseline in wide[col].columns and followup in wide[col].columns:
            out[f"{col}_V{baseline}"] = wide[col][baseline]
            out[f"{col}_V{followup}"] = wide[col][followup]
            out[f"delta_{col}"] = wide[col][followup] - wide[col][baseline]

    return out.reset_index()


def make_delta_roi_df(
    data: pd.DataFrame,
    value_col="GM_roi_ml",
    roi_col="roi",
    subject_col="subject",
    session_col="session",
    baseline=1,
    followup=2,
):
    """
    Delta dataframe for long-format ROI data.
    Output includes one row per subject x ROI.
    """
    df = data[[subject_col, session_col, roi_col, value_col]].copy()

    wide = df.pivot_table(
        index=[subject_col, roi_col],
        columns=session_col,
        values=value_col,
        aggfunc="mean"
    ).reset_index()

    if baseline not in wide.columns or followup not in wide.columns:
        raise ValueError("Baseline/follow-up sessions not found in pivoted ROI data.")

    wide = wide.rename(columns={
        baseline: f"{value_col}_V{baseline}",
        followup: f"{value_col}_V{followup}",
    })

    wide[f"delta_{value_col}"] = (
        wide[f"{value_col}_V{followup}"] - wide[f"{value_col}_V{baseline}"]
    )

    return wide


# ============================================================
# 7) RATIO BUILDERS
# ============================================================

def add_ratio_column(df: pd.DataFrame, numerator_cols, denominator_cols, new_col: str):
    out = df.copy()
    num = out[numerator_cols].sum(axis=1)
    den = out[denominator_cols].sum(axis=1)
    out[new_col] = num / den
    return out


def pivot_roi_long_to_wide(
    data: pd.DataFrame,
    index_cols=("subject", "session", "group"),
    roi_col="roi",
    value_col="GM_roi_ml",
):
    """
    Convert long ROI dataframe to wide ROI matrix so you can build ratios.
    """
    df = data.copy()

    wide = df.pivot_table(
        index=list(index_cols),
        columns=roi_col,
        values=value_col,
        aggfunc="mean"
    ).reset_index()

    wide.columns.name = None
    return wide


def add_standard_remano_ratios(df_wide: pd.DataFrame):
    """
    Add the common ratio metrics from your plan.
    """
    out = df_wide.copy()

    if set(["NAcc_L", "NAcc_R", "caudate_L", "caudate_R", "putamen_L", "putamen_R", "PFC_union"]).issubset(out.columns):
        out["ratio_striatum_pfc"] = (
            out[["NAcc_L", "NAcc_R", "caudate_L", "caudate_R", "putamen_L", "putamen_R"]].sum(axis=1)
            / out["PFC_union"]
        )

    if set(["NAcc_L", "NAcc_R", "OFC_medial"]).issubset(out.columns):
        out["ratio_nacc_ofc"] = (
            out[["NAcc_L", "NAcc_R"]].sum(axis=1) / out["OFC_medial"]
        )

    if set(["caudate_L", "caudate_R", "putamen_L", "putamen_R"]).issubset(out.columns):
        out["ratio_caudate_putamen"] = (
            out[["caudate_L", "caudate_R"]].sum(axis=1)
            / out[["putamen_L", "putamen_R"]].sum(axis=1)
        )

    if set(["caudate_L", "caudate_R", "putamen_L", "putamen_R", "ACC_rostral"]).issubset(out.columns):
        out["ratio_striatum_acc"] = (
            out[["caudate_L", "caudate_R", "putamen_L", "putamen_R"]].sum(axis=1)
            / out["ACC_rostral"]
        )

    if set(["Left_Amygdala", "Right_Amygdala", "Insular_Cortex", "OFC_medial"]).issubset(out.columns):
        out["ratio_limbic_frontal"] = (
            out[["Left_Amygdala", "Right_Amygdala", "Insular_Cortex"]].sum(axis=1)
            / out["OFC_medial"]
        )

    if set(["Hypothalamus", "PFC_union"]).issubset(out.columns):
        out["ratio_hypothalamus_pfc"] = out["Hypothalamus"] / out["PFC_union"]

    if set(["cerebellum_whole", "PFC_union"]).issubset(out.columns):
        out["ratio_cerebellum_pfc"] = out["cerebellum_whole"] / out["PFC_union"]

    if set(["thalamus_L", "thalamus_R", "PFC_union"]).issubset(out.columns):
        out["ratio_thalamus_pfc"] = (
            out[["thalamus_L", "thalamus_R"]].sum(axis=1) / out["PFC_union"]
        )

    if set(["Left_Amygdala", "Right_Amygdala", "ACC_rostral"]).issubset(out.columns):
        out["ratio_amygdala_acc"] = (
            out[["Left_Amygdala", "Right_Amygdala"]].sum(axis=1) / out["ACC_rostral"]
        )

    return out


# ============================================================
# 8) PREDICTIVE MODELING
# ============================================================

def run_prediction(
    data: pd.DataFrame,
    features,
    target: str,
    task="classification",
    model_name="logistic",
    cv=5,
    scoring=None,
    random_state=42,
):
    """
    Generic CV-based predictive modeling wrapper.
    """
    df = data[list(features) + [target]].dropna().copy()
    X = df[list(features)]
    y = df[target]

    if task == "classification":
        if model_name == "logistic":
            model = LogisticRegression(max_iter=5000)
        elif model_name == "rf":
            model = RandomForestClassifier(random_state=random_state)
        else:
            raise ValueError("Unknown classification model.")

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        if scoring is None:
            scoring = "roc_auc"

        cv_obj = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )

    elif task == "regression":
        if model_name == "linear":
            model = LinearRegression()
        elif model_name == "rf":
            model = RandomForestRegressor(random_state=random_state)
        else:
            raise ValueError("Unknown regression model.")

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        if scoring is None:
            scoring = "r2"

        cv_obj = KFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )

    else:
        raise ValueError("task must be 'classification' or 'regression'")

    scores = cross_val_score(pipe, X, y, cv=cv_obj, scoring=scoring)

    return {
        "target": target,
        "task": task,
        "model": model_name,
        "scoring": scoring,
        "n": len(df),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "features": list(features),
    }


# ============================================================
# 9) VOXEL-BASED / VBM HELPERS
# ============================================================

def collect_images(root, group_label=None, pattern="s8mwp1rsub-*_ses-*_T1w.nii*"):
    """
    Collect CAT12-smoothed GM maps with subject/session metadata.
    """
    rows = []
    root = Path(root)

    for p in sorted(root.glob(pattern)):
        subject, session = parse_sub_ses(p.name)
        row = {
            "subject": subject,
            "session": session,
            "path": str(p),
        }
        if group_label is not None:
            row["group"] = group_label
        rows.append(row)

    return pd.DataFrame(rows)


def make_paired_path_table(
    df_imgs: pd.DataFrame,
    group=None,
    subject_col="subject",
    session_col="session",
    path_col="path",
    baseline=1,
    followup=2,
):
    """
    Wide table with one row per subject, V1/V2 paths.
    """
    df = df_imgs.copy()

    if group is not None and "group" in df.columns:
        df = df[df["group"] == group].copy()

    wide = (
        df.pivot(index=subject_col, columns=session_col, values=path_col)
        .dropna(subset=[baseline, followup])
        .sort_index()
    )

    return wide


def build_difference_images(
    wide_paths: pd.DataFrame,
    baseline=1,
    followup=2,
    expr="img2 - img1",
    verbose=True,
):
    """
    Build paired difference maps, e.g. V2 - V1.
    """
    diff_imgs = []
    kept_subjects = []

    for i, (subj, row) in enumerate(wide_paths.iterrows(), start=1):
        path_v1 = row[baseline]
        path_v2 = row[followup]

        img_v1 = load_img(path_v1)
        img_v2 = load_img(path_v2)

        if img_v1.shape != img_v2.shape or not np.allclose(img_v1.affine, img_v2.affine):
            raise ValueError(f"Grid mismatch for subject {subj}")

        diff = math_img(expr, img1=img_v1, img2=img_v2)
        diff_imgs.append(diff)
        kept_subjects.append(subj)

        if verbose:
            print(f"[{i:03d}] subject {subj} done")

    return diff_imgs, kept_subjects


def compute_group_mask_from_imgs(imgs):
    """
    Average image -> brain mask.
    """
    mean_gm = mean_img(imgs)
    mask_img = compute_brain_mask(mean_gm)
    return mask_img


def build_mean_gm_mask_from_paths(paths, threshold=0.3, binarize=True):
    """
    Build a mask from the mean of original GM maps, not difference maps.
    """
    imgs = [load_img(p) for p in paths]
    mean_gm = mean_img(imgs)
    mask_img = math_img("img > thr", img=mean_gm, thr=threshold)

    if binarize:
        mask_img = math_img("img.astype(int)", img=mask_img)

    return mask_img, mean_gm


def run_second_level_one_sample(
    imgs,
    mask_img=None,
    smoothing_fwhm=None,
    output_type="z_score",
):
    """
    One-sample second-level model.
    Good for:
    - AN V2-V1 recovery map
    - HC test-retest V2-V1 map
    """
    if len(imgs) < 5:
        raise ValueError(f"Too few images for second-level model (N={len(imgs)}).")

    design = pd.DataFrame({"intercept": np.ones(len(imgs))})

    slm = SecondLevelModel(mask_img=mask_img, smoothing_fwhm=smoothing_fwhm)
    slm = slm.fit(imgs, design_matrix=design)

    stat_map = slm.compute_contrast("intercept", output_type=output_type)
    return slm, stat_map, design


def run_second_level_two_group(
    df_imgs: pd.DataFrame,
    path_col="path",
    group_col="group",
    covariates=None,
    mask_img=None,
    smoothing_fwhm=None,
    output_type="z_score",
    drop_duplicates_on=("subject", "session"),
    verbose=True,
):
    """
    Cross-sectional voxelwise group comparison.
    Example:
        GM_voxel ~ group + age + TIV_ml
    More robust version:
    - coerces covariates to numeric
    - removes NaN / inf
    - optionally drops duplicate subject-session rows
    - checks design matrix before fitting
    """
    if covariates is None:
        covariates = []

    df = df_imgs.copy()

    needed = [path_col, group_col] + list(covariates)
    needed_existing = [c for c in needed if c in df.columns]

    if drop_duplicates_on is not None:
        dup_cols = [c for c in drop_duplicates_on if c in df.columns]
        if len(dup_cols) > 0:
            n_before = len(df)
            df = df.drop_duplicates(subset=dup_cols).copy()
            if verbose and len(df) != n_before:
                print(f"[run_second_level_two_group] dropped {n_before - len(df)} duplicate rows")

    for c in covariates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed_existing).copy()

    if verbose:
        print("[run_second_level_two_group] N after cleaning:", len(df))
        print("[run_second_level_two_group] missing after cleaning:")
        print(df[needed_existing].isna().sum())

    if len(df) == 0:
        raise ValueError("No rows left after cleaning for second-level two-group model.")

    imgs = df[path_col].tolist()

    design = pd.DataFrame(index=np.arange(len(df)))
    design["intercept"] = 1.0

    grp = df[group_col].astype("category")
    if len(grp.cat.categories) != 2:
        raise ValueError(
            f"Two-group comparison requires exactly 2 group levels, got {list(grp.cat.categories)}"
        )

    design["group"] = (grp == grp.cat.categories[1]).astype(int).values

    for c in covariates:
        if c not in df.columns:
            raise ValueError(f"Missing covariate column: {c}")
        design[c] = pd.to_numeric(df[c], errors="coerce").values

    design = design.replace([np.inf, -np.inf], np.nan)

    bad_mask = design.isna().any(axis=1)
    if bad_mask.any():
        if verbose:
            print("[run_second_level_two_group] dropping rows with invalid design values:")
            print(
                pd.concat(
                    [
                        df.loc[bad_mask, [group_col] + [c for c in covariates if c in df.columns]].reset_index(drop=True),
                        design.loc[bad_mask].reset_index(drop=True),
                    ],
                    axis=1,
                )
            )
        keep_mask = ~bad_mask
        design = design.loc[keep_mask].reset_index(drop=True)
        df = df.loc[keep_mask].reset_index(drop=True)
        imgs = df[path_col].tolist()

    if design.isna().any().any():
        raise ValueError("Design matrix still contains NaN after cleaning.")

    if np.isinf(design.to_numpy(dtype=float)).any():
        raise ValueError("Design matrix still contains inf after cleaning.")

    if verbose:
        print("[run_second_level_two_group] design columns:", design.columns.tolist())
        print("[run_second_level_two_group] group counts:")
        print(df[group_col].value_counts())
        print("[run_second_level_two_group] design head:")
        print(design.head())

    slm = SecondLevelModel(mask_img=mask_img, smoothing_fwhm=smoothing_fwhm)
    slm = slm.fit(imgs, design_matrix=design)

    stat_map = slm.compute_contrast("group", output_type=output_type)
    return slm, stat_map, design


def run_second_level_with_design(
    imgs,
    design_matrix: pd.DataFrame,
    contrast,
    mask_img=None,
    smoothing_fwhm=None,
    output_type="z_score",
):
    """
    Most flexible wrapper when you already built the design matrix yourself.
    Useful for:
    - group * session
    - remission * session
    - subtype * session
    - BMI maps
    - duration maps
    """
    if len(imgs) != len(design_matrix):
        raise ValueError("Mismatch between number of images and design rows.")

    slm = SecondLevelModel(mask_img=mask_img, smoothing_fwhm=smoothing_fwhm)
    slm = slm.fit(imgs, design_matrix=design_matrix)

    stat_map = slm.compute_contrast(contrast, output_type=output_type)
    return slm, stat_map


def threshold_map(
    stat_map,
    alpha=0.05,
    height_control="fdr",
    cluster_threshold=0,
    two_sided=True,
):
    """
    FDR/FPR thresholding wrapper.
    """
    thr_img, thr = threshold_stats_img(
        stat_map,
        alpha=alpha,
        height_control=height_control,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
    )
    return thr_img, thr


def save_nifti(img, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(str(out_path))
    return out_path


def threshold_and_save_map(
    stat_map,
    out_dir,
    base_name,
    alpha=0.05,
    height_control="fdr",
    two_sided=True,
):
    """
    Threshold a stat map and save both raw and thresholded versions.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = save_nifti(stat_map, out_dir / f"{base_name}_zmap.nii.gz")

    thr_img, thr = threshold_map(
        stat_map,
        alpha=alpha,
        height_control=height_control,
        two_sided=two_sided,
    )

    thr_path = save_nifti(
        thr_img,
        out_dir / f"{base_name}_zmap_FDR{int(alpha*100):02d}.nii.gz"
    )

    return {
        "z_map": stat_map,
        "z_thr_map": thr_img,
        "threshold": thr,
        "raw_path": raw_path,
        "thr_path": thr_path,
    }


def split_signed_map(stat_img):
    """
    Split a thresholded statistical map into positive and negative maps.
    """
    pos_img = math_img("img * (img > 0)", img=stat_img)
    neg_img = math_img("img * (img < 0)", img=stat_img)
    return pos_img, neg_img


def save_split_signed_map(stat_img, out_dir, base_name):
    """
    Split a thresholded map into positive/negative components and save both.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_img, neg_img = split_signed_map(stat_img)

    pos_path = save_nifti(pos_img, out_dir / f"{base_name}_increases.nii.gz")
    neg_path = save_nifti(neg_img, out_dir / f"{base_name}_decreases.nii.gz")

    return {
        "positive_img": pos_img,
        "negative_img": neg_img,
        "positive_path": pos_path,
        "negative_path": neg_path,
    }


def extract_clusters_table(
    stat_img,
    stat_threshold=1e-6,
    cluster_threshold=1000,
    two_sided=True,
):
    """
    Extract cluster table from a thresholded statistical map.
    """
    return get_clusters_table(
        stat_img=stat_img,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
    )


def run_nonparametric_one_sample(
    imgs,
    mask_img=None,
    n_perm=5000,
    two_sided_test=False,
    tfce=True,
    smoothing_fwhm=None,
    n_jobs=1,
):
    """
    Permutation one-sample inference on paired difference maps.
    """
    design = pd.DataFrame({"intercept": np.ones(len(imgs))})

    out = non_parametric_inference(
        second_level_input=imgs,
        design_matrix=design,
        second_level_contrast="intercept",
        mask=mask_img,
        model_intercept=False,
        n_perm=n_perm,
        two_sided_test=two_sided_test,
        tfce=tfce,
        smoothing_fwhm=smoothing_fwhm,
        n_jobs=n_jobs,
    )
    return out, design


def build_cross_sectional_vbm_dataset(
    df_paths: pd.DataFrame,
    df_covars: pd.DataFrame,
    on=("subject", "session"),
    path_col="path",
):
    """
    Merge image paths with subject/session covariates for voxelwise OLS.
    """
    df = df_paths.merge(df_covars, on=list(on), how="inner")
    df = df.dropna(subset=[path_col]).copy()
    return df


def prepare_longitudinal_diff_dataset(
    df_paths: pd.DataFrame,
    df_covars: pd.DataFrame,
    group=None,
    baseline=1,
    followup=2,
):
    """
    Build subject-level paired difference maps plus subject-level covariates.
    Useful for:
    - delta BMI voxelwise map
    - AN remission paired analyses
    """
    wide = make_paired_path_table(
        df_paths,
        group=group,
        baseline=baseline,
        followup=followup
    )
    diff_imgs, subjects = build_difference_images(
        wide,
        baseline=baseline,
        followup=followup,
        verbose=False
    )

    df_subj = df_covars.copy()
    if group is not None and "group" in df_subj.columns:
        df_subj = df_subj[df_subj["group"] == group].copy()

    df_subj = df_subj.drop_duplicates(subset=["subject"]).copy()
    df_subj = df_subj[df_subj["subject"].isin(subjects)].copy()
    df_subj = df_subj.set_index("subject").loc[subjects].reset_index()

    return diff_imgs, df_subj


def summarize_vbm_map_in_rois(
    z_map,
    z_thr_map,
    roi_dir,
    roi_pattern="*.nii*",
):
    """
    Summarize raw and thresholded voxelwise maps within a directory of ROI masks.
    """
    roi_dir = Path(roi_dir)
    rows = []

    for rp in sorted(roi_dir.glob(roi_pattern)):
        roi = load_img(rp)

        roi_rs = resample_to_img(roi, z_map, interpolation="nearest")
        roi_bin = math_img("img > 0", img=roi_rs)

        n_vox = int(roi_bin.get_fdata().sum())
        if n_vox == 0:
            continue

        z_vals = apply_mask(z_map, roi_bin)
        zsig_vals = apply_mask(z_thr_map, roi_bin)

        sig_mask = (zsig_vals != 0)
        sig_pos = (zsig_vals > 0)
        sig_neg = (zsig_vals < 0)

        rows.append({
            "roi_file": rp.name,
            "n_vox": int(z_vals.size),
            "mean_z": float(np.mean(z_vals)),
            "median_z": float(np.median(z_vals)),
            "pct_FDR_sig_voxels": float(sig_mask.mean() * 100.0),
            "mean_z_sig_voxels": float(np.mean(zsig_vals[sig_mask])) if sig_mask.any() else np.nan,
            "mean_z_nonsig_voxels": float(np.mean(z_vals[~sig_mask])) if (~sig_mask).any() else np.nan,
            "pct_sig_pos": float(sig_pos.mean() * 100.0),
            "pct_sig_neg": float(sig_neg.mean() * 100.0),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("pct_FDR_sig_voxels", ascending=False).reset_index(drop=True)

    return out


def run_paired_vbm_one_sample(
    df_imgs,
    group,
    baseline=1,
    followup=2,
    expr="img2 - img1",
    mask_threshold=0.3,
    smoothing_fwhm=None,
    output_type="z_score",
    verbose=True,
):
    """
    Run paired one-sample VBM on within-subject difference images.
    """
    if verbose:
        print("[run_paired_vbm_one_sample] building paired table")

    wide = make_paired_path_table(
        df_imgs=df_imgs,
        group=group,
        baseline=baseline,
        followup=followup,
    )

    if verbose:
        print("[run_paired_vbm_one_sample] wide shape:", wide.shape)
        print(wide.head())

    if wide.empty:
        raise ValueError("No paired subjects found.")

    if verbose:
        print("[run_paired_vbm_one_sample] building difference images")

    diff_imgs, subjects = build_difference_images(
        wide_paths=wide,
        baseline=baseline,
        followup=followup,
        expr=expr,
        verbose=verbose,
    )

    if verbose:
        print("[run_paired_vbm_one_sample] n diff imgs:", len(diff_imgs))

    all_paths = wide[baseline].tolist() + wide[followup].tolist()

    if verbose:
        print("[run_paired_vbm_one_sample] building mean GM mask")

    mask_img, mean_gm = build_mean_gm_mask_from_paths(
        all_paths,
        threshold=mask_threshold,
        binarize=True,
    )

    n_mask = int(mask_img.get_fdata().sum())
    if verbose:
        print("[run_paired_vbm_one_sample] mask voxels:", n_mask)

    if n_mask == 0:
        raise ValueError("Mask is empty.")

    if verbose:
        print("[run_paired_vbm_one_sample] fitting one-sample model")

    slm, z_map, design = run_second_level_one_sample(
        imgs=diff_imgs,
        mask_img=mask_img,
        smoothing_fwhm=smoothing_fwhm,
        output_type=output_type,
    )

    return {
        "wide": wide,
        "subjects": subjects,
        "diff_imgs": diff_imgs,
        "mask_img": mask_img,
        "mean_gm": mean_gm,
        "slm": slm,
        "z_map": z_map,
        "design": design,
    }


# ============================================================
# 10) COMMON DESIGN BUILDERS FOR VBM
# ============================================================

def design_one_sample(n):
    return pd.DataFrame({"intercept": np.ones(n)})


def design_from_dataframe(
    df: pd.DataFrame,
    intercept=True,
    categorical=None,
    continuous=None,
    drop_first=True,
):
    """
    Build design matrix from a dataframe.
    Example:
        categorical=["group", "session"]
        continuous=["age", "TIV_ml", "BMI"]
    """
    if categorical is None:
        categorical = []
    if continuous is None:
        continuous = []

    parts = []

    if intercept:
        parts.append(pd.DataFrame({"intercept": np.ones(len(df))}, index=df.index))

    for col in categorical:
        dummies = pd.get_dummies(df[col].astype("category"), prefix=col, drop_first=drop_first)
        parts.append(dummies)

    for col in continuous:
        parts.append(pd.DataFrame({col: df[col].astype(float).values}, index=df.index))

    design = pd.concat(parts, axis=1)
    return design


# ============================================================
# 11) EXAMPLE FORMULA TEMPLATES
# ============================================================

FORMULAS = {
    "group_baseline": "{Y} ~ C(group) + age + TIV_ml",
    "group_followup": "{Y} ~ C(group) + age + TIV_ml",
    "longitudinal_all": "{Y} ~ C(session) + age + TIV_ml",
    "group_session": "{Y} ~ C(group) * C(session) + age + TIV_ml",
    "hc_retest": "{Y} ~ C(session) + age + TIV_ml",
    "an_recovery": "{Y} ~ C(session) + age + TIV_ml",
    "remission_baseline": "{Y} ~ C(remission_v3) + age + TIV_ml",
    "remission_followup": "{Y} ~ C(remission_v3) + age + TIV_ml",
    "remission_session": "{Y} ~ C(remission_v3) * C(session) + age + TIV_ml",
    "bmi_cross": "{Y} ~ BMI + age + TIV_ml",
    "bmi_group": "{Y} ~ C(group) + BMI + age + TIV_ml",
    "bmi_session": "{Y} ~ BMI * C(session) + age + TIV_ml",
    "duration": "{Y} ~ duree_troubles + age + TIV_ml",
    "age_onset": "{Y} ~ age_at_onset + age + TIV_ml",
    "duration_session": "{Y} ~ duree_troubles * C(session) + age + TIV_ml",
    "anxiety": "{Y} ~ hads_anxiete + age + TIV_ml",
    "depression": "{Y} ~ hads_depression + age + TIV_ml",
    "ocd_obs": "{Y} ~ ybs_obs + age + TIV_ml",
    "ocd_comp": "{Y} ~ ybs_comp + age + TIV_ml",
    "trauma": "{Y} ~ ctq_score + age + TIV_ml",
    "autism": "{Y} ~ raads_score + age + TIV_ml",
    "subtype": "{Y} ~ C(soustype) + age + TIV_ml",
    "subtype_session": "{Y} ~ C(soustype) * C(session) + age + TIV_ml",
}
