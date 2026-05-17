"""
methylation_outlier_detection.py
=================================
Identifies aberrant DNA methylation at CpG islands from phased long-read sequencing data.

Overview
--------
For each CpG island in the candidate file the script:
  1. Tiles the island into overlapping sliding windows.
  2. Computes per-window methylation frequencies for every sample at allele
     level (phase_0 / phase_1 separately) and at combined (diploid) level.
  3. Builds a cohort background distribution per window (mean, SD, IQR).
  4. Tests each sample/window for deviation from the background using a
     two-sided binomial test, then applies Benjamini-Hochberg FDR correction.
  5. Applies configurable deviation filters (SD-based, absolute, or both) and
     an IQR-stability guard to suppress calls in highly variable regions.
  6. Summarises significant window-level hits back to island level, retaining
     islands that exceed the minimum outlier-window count and fraction.

Sex chromosomes are processed separately to avoid distortion of the background
caused by X-inactivation and the absence of chrY in females.

Input
-----
candidate_file : tab-separated, columns: chrom, start, end
directory      : directory containing phased DSS files
                 (*.phase_0.tbx.DSS.txt.gz / *.phase_1.tbx.DSS.txt.gz)
sex_file       : tab-separated, columns: sample_id, sex (M/F)

Output
------
haplotype_results.txt   – allele-level island outliers
combined_results.txt – combined (diploid) island outliers
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pysam
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------
candidate_file = ""          # tab-separated: chrom, start, end
directory      = Path("")    # directory of phased DSS files
sex_file       = "sample_sex.txt"  # tab-separated: sample_id, sex (M/F)

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
# Threshold mode: "sd" | "abs" | "both"
#   "sd"   – flag windows where |deviation| > SD_MULTIPLIER × background SD
#   "abs"  – flag windows where |deviation| > absolute threshold
#   "both" – both conditions must be satisfied simultaneously
THRESHOLD_MODE = "both"

MIN_COVERAGE              = 10    # minimum reads at a CpG site to include it
SD_MULTIPLIER             = 3     # multiplier for the SD-based deviation threshold
FDR_THRESHOLD             = 0.05  # Benjamini-Hochberg adjusted p-value cutoff
MIN_CPG_SITES             = 10    # minimum covered CpG sites required per window
MIN_CPG_COVERAGE_FRACTION = 0.5   # minimum fraction of CpG sites covered per window
WINDOW_SIZE_BP            = 300   # sliding window width in base pairs
WINDOW_STEP_BP            = 150   # sliding window step size in base pairs
MIN_OUTLIER_WINDOWS       = 1     # minimum outlier windows required to report an island
MIN_OUTLIER_FRACTION      = 0.3   # minimum fraction of tested windows that must be outliers

# IQR-based outlier thresholds
IQR_DEV_THRESHOLD = 1.5    # minimum IQR-scaled deviation to call a window an outlier
IQR_MAX_THRESHOLD = 0.25   # suppress calls if the background IQR exceeds this value

# Absolute deviation thresholds
ABS_DEV_THRESHOLD_ALLELE   = 0.40  # used for allele-level (phased) analysis
ABS_DEV_THRESHOLD_COMBINED = 0.25  # used for combined (diploid) analysis

def get_cpg_islands(cpg_file):
    """
    Load CpG island regions from a tab-delimited file.

    Returns a deduplicated DataFrame with columns: chrom, start, end, location.
    ``location`` is formatted as ``chrom:start-end`` and serves as a unique
    region identifier throughout the pipeline.
    """
    cpg_df     = pd.read_csv(cpg_file, sep="\t", skipinitialspace=True)
    cpg_islands = cpg_df[["chrom", "start", "end"]].copy()
    cpg_islands["location"] = (
        cpg_islands["chrom"] + ":" +
        cpg_islands["start"].astype(str) + "-" +
        cpg_islands["end"].astype(str)
    )
    return cpg_islands.drop_duplicates().sort_values(["chrom", "start"])


def get_sample_sex(sex_file):
    """
    Load sample sex annotations from *sex_file*.

    Returns
    -------
    males, females : tuple of sets
        Sets of sample IDs annotated as male (M) or female (F).

    Raises
    ------
    ValueError
        If *sex_file* does not exist or is empty.
    """
    path = Path(sex_file)
    if not path.is_file():
        raise ValueError(f"sex_file not found: {sex_file}")

    sex_df = pd.read_csv(path, sep="\t")
    if sex_df.empty:
        raise ValueError("sex_df is empty — check sample_sex.txt")

    males   = set(sex_df[sex_df["sex"] == "M"]["sample_id"])
    females = set(sex_df[sex_df["sex"] == "F"]["sample_id"])
    print(f"Sex annotations loaded — Males: {len(males)}, Females: {len(females)}")
    return males, females


def split_cpg_by_chrom(cpg_df):
    """
    Partition CpG islands into autosomal, chrX, and chrY subsets.

    Sex chromosomes are processed separately to prevent distortion of the
    cohort background distribution.

    Returns
    -------
    autosomal, x_cpg, y_cpg : tuple of DataFrames
    """
    autosomal = cpg_df[~cpg_df["chrom"].isin(["chrX", "chrY"])]
    x_cpg     = cpg_df[cpg_df["chrom"] == "chrX"]
    y_cpg     = cpg_df[cpg_df["chrom"] == "chrY"]
    return autosomal, x_cpg, y_cpg

  
def build_windows_df(cpg_df, window_size=WINDOW_SIZE_BP, step=WINDOW_STEP_BP):
    """
    Generate sliding windows of fixed bp size within each CpG island.

    Islands smaller than *window_size* are represented by a single window
    covering the whole island. Each window records a ``parent_island`` field
    linking it back to its source island for aggregation after outlier testing.

    Parameters
    ----------
    cpg_df      : DataFrame from `get_cpg_islands`
    window_size : window width in base pairs
    step        : step size between consecutive window starts

    Returns
    -------
    DataFrame with columns: chrom, start, end, location, parent_island.
    """
    all_windows = []
    for _, region in cpg_df.iterrows():
        island_size = region["end"] - region["start"]
        if island_size <= window_size:
            all_windows.append({
                "chrom":         region["chrom"],
                "start":         region["start"],
                "end":           region["end"],
                "location":      f"{region['chrom']}:{region['start']}-{region['end']}",
                "parent_island": region["location"],
            })
            continue
        pos = region["start"]
        while pos < region["end"]:
            win_end = min(pos + window_size, region["end"])
            all_windows.append({
                "chrom":         region["chrom"],
                "start":         pos,
                "end":           win_end,
                "location":      f"{region['chrom']}:{pos}-{win_end}",
                "parent_island": region["location"],
            })
            pos += step

    windows_df = pd.DataFrame(all_windows).reset_index(drop=True)
    print(
        f"Generated {len(windows_df)} windows from {len(cpg_df)} CpG islands "
        f"(window={window_size} bp, step={step} bp)"
    )
    return windows_df


def file_to_tabix(filepath):
    """
    Open a tabix-indexed DSS file and derive a sample identifier from its name.

    Expected filename format: ``{sample}.{...}.{...}.{phase}.tbx.DSS.txt.gz``
    The identifier is constructed as ``{sample}_{phase}``.

    Returns
    -------
    tbx        : pysam.TabixFile
    identifier : str
    """
    tbx        = pysam.TabixFile(str(filepath))
    parts      = Path(filepath).stem.split(".")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filepath.name}")
    identifier = f"{parts[0]}_{parts[3]}"
    return tbx, identifier


def fetch_chrom_records(sample_tabix, chrom):
    """
    Fetch and parse all tabix records for one chromosome.

    Uses the pandas CSV parser.

    Returns
    -------
    DataFrame with columns: chrom, start, end, meth_reads, total_reads,
    or ``None`` if the chromosome has no records.
    """
    lines = sample_tabix.fetch(chrom, multiple_iterators=False)
    buf   = io.BytesIO("\n".join(lines).encode())
    if buf.getbuffer().nbytes == 0:
        return None
    return pd.read_csv(
        buf,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "meth_reads", "total_reads"],
        dtype={"chrom": str, "start": int, "end": int,
               "meth_reads": int, "total_reads": int},
    )


def aggregate_windows(chrom_df, chrom_windows):
    """
    Aggregate per-CpG read counts into per-window totals.

    Uses `numpy.searchsorted` on sorted CpG positions to locate,
    the slice of CpG sites that fall within each window, then sums
    their read counts directly.

    CpG sites with fewer than MIN_COVERAGE reads are excluded from the covered
    totals (``n_cpg_sites``, ``total_reads``, ``meth_reads``) but counted in
    ``n_cpg_sites_total`` for the coverage-fraction filter downstream.

    Parameters
    ----------
    chrom_df      : per-chromosome CpG records from `fetch_chrom_records`
    chrom_windows : subset of windows_df for this chromosome

    Returns
    -------
    DataFrame indexed by window location with columns:
    location, total_reads, meth_reads, n_cpg_sites, n_cpg_sites_total.
    """
    chrom_df_covered = chrom_df[chrom_df["total_reads"] >= MIN_COVERAGE]

    all_sorted = chrom_df.sort_values("start").drop_duplicates("start")
    cov_sorted = chrom_df_covered.sort_values("start").drop_duplicates("start")

    all_pos   = all_sorted["start"].values
    cov_pos   = cov_sorted["start"].values
    cov_meth  = cov_sorted["meth_reads"].values
    cov_total = cov_sorted["total_reads"].values

    win_starts = chrom_windows["start"].values
    win_ends   = chrom_windows["end"].values

    lo_all = np.searchsorted(all_pos, win_starts, side="left")
    hi_all = np.searchsorted(all_pos, win_ends,   side="left")
    lo_cov = np.searchsorted(cov_pos, win_starts, side="left")
    hi_cov = np.searchsorted(cov_pos, win_ends,   side="left")

    n = len(chrom_windows)
    n_cpg_total = np.zeros(n, dtype=np.int64)
    n_cpg_cov   = np.zeros(n, dtype=np.int64)
    total_reads = np.zeros(n, dtype=np.int64)
    meth_reads  = np.zeros(n, dtype=np.int64)

    for i in range(n):
        n_cpg_total[i] = hi_all[i] - lo_all[i]
        n_cpg_cov[i]   = hi_cov[i] - lo_cov[i]
        total_reads[i] = cov_total[lo_cov[i]:hi_cov[i]].sum()
        meth_reads[i]  = cov_meth[lo_cov[i]:hi_cov[i]].sum()

    return pd.DataFrame({
        "location":          chrom_windows["location"].values,
        "total_reads":       total_reads,
        "meth_reads":        meth_reads,
        "n_cpg_sites":       n_cpg_cov,
        "n_cpg_sites_total": n_cpg_total,
    })


def empty_chrom_stats(chrom_windows):
    """Return a zeroed stats DataFrame for a chromosome with no coverage."""
    return pd.DataFrame({
        "location":          chrom_windows["location"].values,
        "total_reads":       0,
        "meth_reads":        0,
        "n_cpg_sites":       0,
        "n_cpg_sites_total": 0,
    })


def calc_meth_freq(sample_tabix, windows_df):
    """
    Compute per-window methylation frequencies for a single sample.

    Iterates over chromosomes, fetches CpG records via tabix, and aggregates
    read counts into windows. Chromosomes absent from the tabix index are filled
    with zeros.

    Returns
    -------
    meth_freq : Series indexed by window location (NaN where total_reads == 0)
    stats     : DataFrame with raw read counts per window
    """
    available_contigs = set(sample_tabix.contigs)
    all_stats = []
    for chrom, chrom_windows in windows_df.groupby("chrom", sort=False):
        if chrom not in available_contigs:
            all_stats.append(empty_chrom_stats(chrom_windows))
            continue
        chrom_df = fetch_chrom_records(sample_tabix, chrom)
        if chrom_df is None:
            all_stats.append(empty_chrom_stats(chrom_windows))
            continue
        all_stats.append(aggregate_windows(chrom_df, chrom_windows))

    stats     = pd.concat(all_stats).set_index("location")
    meth_freq = stats["meth_reads"] / stats["total_reads"].replace(0, np.nan)
    return meth_freq, stats


def compute_background_stats(background_df):
    """
    Compute cohort-level summary statistics across sample columns.

    Adds ``mean``, ``sd``, ``q25``, ``q75``, and ``iqr`` columns to
    *background_df* in-place (computed row-wise, ignoring NaN values).

    Returns
    -------
    The modified DataFrame.
    """
    numeric_cols = background_df.select_dtypes(include=[float]).columns
    bg_matrix    = background_df[numeric_cols].values

    with np.errstate(all="ignore"):
        background_df["mean"] = np.nanmean(bg_matrix,            axis=1)
        background_df["sd"]   = np.nanstd(bg_matrix,             axis=1)
        background_df["q25"]  = np.nanpercentile(bg_matrix, 25,  axis=1)
        background_df["q75"]  = np.nanpercentile(bg_matrix, 75,  axis=1)
    background_df["iqr"] = background_df["q75"] - background_df["q25"]
    return background_df


def process_files_separately(directory, windows_df, samples=None):
    """
    Process each phased DSS file independently (allele-level analysis).

    Opens each ``*.DSS.txt.gz`` file found in *directory*, computes per-window
    methylation frequencies, and assembles a cohort background DataFrame.

    Parameters
    ----------
    directory  : Path to the directory containing phased DSS files
    windows_df : sliding windows DataFrame from `build_windows_df`
    samples    : optional set of sample base names to restrict processing;
                 if None all files in *directory* are processed

    Returns
    -------
    background_df : DataFrame with cohort background statistics per window
    sample_cache  : dict of {identifier: (meth_freq, stats)}
    raw_cache     : dict of {file.stem: (meth_freq, stats)} — used by
                    `process_files_together` to avoid re-reading files
    """
    background_df = windows_df[["location"]].set_index("location")
    sample_cache  = {}
    raw_cache     = {}

    files = [
        f for f in sorted(directory.glob("*.DSS.txt.gz"))
        if not f.name.startswith(".")
        and (samples is None or f.stem.split(".")[0] in samples)
    ]

    if samples is not None:
        found   = {f.stem.split(".")[0] for f in files}
        missing = samples - found
        if missing:
            print(f"WARNING: {len(missing)} sample(s) have no matching files: {missing}")

    sample_series = []
    for filepath in files:
        sample_tbx, identifier = file_to_tabix(filepath)
        meth_freq, stats       = calc_meth_freq(sample_tbx, windows_df)
        sample_tbx.close()
        meth_freq.name = identifier
        print(f"    {identifier}")

        sample_series.append(meth_freq)
        sample_cache[identifier]    = (meth_freq, stats)
        raw_cache[filepath.stem]    = (meth_freq, stats)

    if sample_series:
        background_df = pd.concat([background_df] + sample_series, axis=1)
    return compute_background_stats(background_df), sample_cache, raw_cache


def process_files_together(directory, windows_df, raw_cache, samples=None):
    """
    Combine phase_0 and phase_1 allele data into diploid methylation frequencies.

    Read counts from both phases are summed to produce a diploid total per
    window. Windows where either phase has zero coverage, or where fewer than
    MIN_CPG_SITES CpG sites are covered in either phase, are masked to zero.
    ``n_cpg_sites`` is taken as the maximum across phases to avoid
    double-counting sites present in both.

    Pre-computed allele data from `process_files_separately` (via
    *raw_cache*) is reused so no files are re-opened.

    Parameters
    ----------
    directory  : Path used to enumerate phase_0/phase_1 filename pairs
    windows_df : sliding windows DataFrame from `build_windows_df`
    raw_cache  : dict of {file.stem: (meth_freq, stats)} returned by
                `process_files_separately`
    samples    : optional set of sample base names to restrict processing

    Returns
    -------
    background_df : DataFrame with cohort background statistics per window
    sample_cache  : dict of {identifier: (meth_freq, combined_stats)};
                    identifiers are formatted as ``{sample}_combined``
    """
    background_df = windows_df[["location"]].set_index("location")
    sample_series = []
    sample_cache  = {}

    for file0 in sorted(directory.glob("*.phase_0.tbx.DSS.txt.gz")):
        file1 = Path(str(file0).replace(".phase_0.", ".phase_1."))
        if not file1.is_file():
            print(f"WARNING: missing phase_1 file for {file0.name}, skipping")
            continue

        base_name = file0.stem.split(".")[0]
        if samples is not None and base_name not in samples:
            continue
        if file0.stem not in raw_cache or file1.stem not in raw_cache:
            print(f"WARNING: {base_name} missing from raw_cache, skipping")
            continue

        identifier = f"{base_name}_combined"
        print(f"    {identifier}")

        meth_freq0, stats0 = raw_cache[file0.stem]
        meth_freq1, stats1 = raw_cache[file1.stem]

        # Mask windows where either phase lacks sufficient CpG coverage
        both_phases_cpg_mask = (
            (stats0["n_cpg_sites"] >= MIN_CPG_SITES) &
            (stats1["n_cpg_sites"] >= MIN_CPG_SITES)
        )
        mask = (
            (stats0["total_reads"] != 0) &
            (stats1["total_reads"] != 0) &
            both_phases_cpg_mask
        )

        combined_stats = stats0.add(stats1).where(mask, 0)
        combined_stats["n_cpg_sites"] = (
            stats0["n_cpg_sites"]
            .combine(stats1["n_cpg_sites"], max)
            .where(mask, 0)
        )
        combined_stats["n_cpg_sites_total"] = (
            stats0["n_cpg_sites_total"]
            .combine(stats1["n_cpg_sites_total"], max)
        )

        meth_freq = (
            combined_stats["meth_reads"] /
            combined_stats["total_reads"].replace(0, np.nan)
        )
        meth_freq.name = identifier
        sample_series.append(meth_freq)
        sample_cache[identifier] = (meth_freq, combined_stats)

    if sample_series:
        background_df = pd.concat([background_df] + sample_series, axis=1)
    return compute_background_stats(background_df), sample_cache


def binom_test(meth_reads, total_reads, expected_p):
    """
    Vectorised two-sided binomial test for deviation from expected methylation.

    Tests whether the observed methylation count in each window deviates
    significantly from the cohort background mean. Windows with 
    NaN expected_p receive a p-value of 1.

    Parameters
    ----------
    meth_reads  : array-like of observed methylated read counts
    total_reads : array-like of total read counts
    expected_p  : array-like of expected methylation probabilities (background mean)

    Returns
    -------
    p_values : ndarray, one p-value per window
    """
    meth_reads  = np.asarray(meth_reads,  dtype=float)
    total_reads = np.asarray(total_reads, dtype=float)
    expected_p  = np.asarray(expected_p,  dtype=float)

    valid    = ~np.isnan(expected_p)
    p_values = np.ones(len(total_reads))

    m = meth_reads[valid].astype(int)
    n = total_reads[valid].astype(int)
    p = expected_p[valid]

    lower = binom.cdf(m, n, p)
    upper = 1 - binom.cdf(m - 1, n, p)
    p_values[valid] = 2 * np.minimum(np.minimum(lower, upper), 0.5)
    return p_values


def compute_outliers(sample_freq, sample_stats, background_df, identifier,
                     abs_threshold=None):
    """
    Compute per-window deviation statistics for one sample vs. the background.

    Pre-filters applied before outlier scoring:
    * Windows with zero total coverage are excluded.
    * Windows where covered CpG sites are fewer than MIN_CPG_COVERAGE_FRACTION
      of total sites are excluded (avoids false positives from partial windows).
    * Windows with fewer than MIN_CPG_SITES covered CpG sites are excluded.

    FDR correction and final thresholding are applied downstream in
    `test_sample_meth`.

    Parameters
    ----------
    sample_freq   : Series of per-window methylation frequencies for the sample
    sample_stats  : DataFrame of raw read counts for the sample
    background_df : cohort background DataFrame with mean, sd, iqr columns
    identifier    : sample identifier string
    abs_threshold : absolute deviation threshold (required when THRESHOLD_MODE
                    is "abs" or "both")

    Returns
    -------
    DataFrame of windows passing pre-filters, with deviation statistics and
    raw p-values.
    """
    sample_filtered   = sample_freq.reindex(background_df.index)
    stats_indexed     = sample_stats.reindex(background_df.index).fillna(0)

    total_reads       = stats_indexed["total_reads"]
    meth_reads        = stats_indexed["meth_reads"]
    n_cpg_sites       = stats_indexed["n_cpg_sites"]
    n_cpg_sites_total = stats_indexed["n_cpg_sites_total"]
    expected_p        = background_df["mean"]

    p_values  = binom_test(meth_reads, total_reads, expected_p)
    deviation = sample_filtered.values - background_df["mean"].values
    abs_dev   = np.abs(deviation)

    with np.errstate(divide="ignore", invalid="ignore"):
        iqr_dev = np.where(
            background_df["iqr"].values > 0,
            abs_dev / background_df["iqr"].values,
            np.inf,
        )

    sample_outliers_df = pd.DataFrame({
        "CpG_island":        sample_filtered.index,
        "sample":            identifier,
        "total_reads":       total_reads.values,
        "sample_meth":       sample_filtered.values,
        "n_cpg_sites":       n_cpg_sites.values,
        "n_cpg_sites_total": n_cpg_sites_total.values,
        "mean":              background_df["mean"].values,
        "sd":                background_df["sd"].values,
        "q25":               background_df["q25"].values,
        "q75":               background_df["q75"].values,
        "iqr":               background_df["iqr"].values,
        "iqr_dev":           iqr_dev,
        "deviation":         deviation,
        "abs_deviation":     abs_dev,
        "p_value":           p_values,
    })

    coverage_fraction = np.where(
        sample_outliers_df["n_cpg_sites_total"] > 0,
        sample_outliers_df["n_cpg_sites"] / sample_outliers_df["n_cpg_sites_total"],
        0,
    )
    valid_mask = (
        (sample_outliers_df["total_reads"] > 0) &
        (coverage_fraction >= MIN_CPG_COVERAGE_FRACTION) &
        (sample_outliers_df["n_cpg_sites"] >= MIN_CPG_SITES)
    )
    return sample_outliers_df[valid_mask]


def apply_fdr(all_outliers_df):
    """
    Apply Benjamini-Hochberg FDR correction to the pooled outlier DataFrame.

    Corrects p-values across all samples and windows simultaneously. Adds
    ``p_adj`` and ``significant`` columns to a copy of the input.

    Returns
    -------
    DataFrame with p_adj and significant columns added.
    """
    reject, p_adj, _, _ = multipletests(all_outliers_df["p_value"], method="fdr_bh")
    all_outliers_df = all_outliers_df.copy()
    all_outliers_df["p_adj"]       = p_adj
    all_outliers_df["significant"] = reject
    return all_outliers_df


def build_deviation_mask(df, abs_threshold):
    """
    Return a boolean mask for windows passing the configured deviation threshold.

    Behaviour is controlled by THRESHOLD_MODE:
        ``"sd"``   – |deviation| > SD_MULTIPLIER × background SD
        ``"abs"``  – |deviation| > abs_threshold
        ``"both"`` – both SD and absolute conditions must hold simultaneously

    Raises
    ------
    ValueError
        If *abs_threshold* is None when THRESHOLD_MODE requires it, or if
        THRESHOLD_MODE is not one of the recognised values.
    """
    sd_pass = df["abs_deviation"].fillna(0) > SD_MULTIPLIER * df["sd"]

    if THRESHOLD_MODE == "sd":
        return sd_pass

    if abs_threshold is None:
        raise ValueError(
            f"abs_threshold must be provided when THRESHOLD_MODE is {THRESHOLD_MODE!r}"
        )
    abs_pass = df["abs_deviation"].fillna(0) > abs_threshold

    if THRESHOLD_MODE == "abs":
        return abs_pass
    if THRESHOLD_MODE == "both":
        return sd_pass & abs_pass

    raise ValueError(
        f"Unknown THRESHOLD_MODE: {THRESHOLD_MODE!r}. "
        "Expected 'sd', 'abs', or 'both'."
    )


def test_sample_meth(sample_cache, background_df, abs_threshold=None):
    """
    Test all samples in *sample_cache* for per-window methylation outliers.

    Steps:
        1. Compute per-window deviation statistics for each sample.
        2. Pool candidate windows across all samples.
        3. Apply Benjamini-Hochberg FDR correction over the pooled p-values.
        4. Retain windows passing FDR threshold, deviation threshold,
           IQR-stability guard, and minimum CpG-site count.

    Returns
    -------
    DataFrame of significant outlier windows for summarisation by
    `summarise_to_islands`, or an empty DataFrame if none are found.
    """
    all_outliers = [
        compute_outliers(
            sample_freq, sample_stats, background_df,
            identifier, abs_threshold=abs_threshold,
        )
        for identifier, (sample_freq, sample_stats) in sample_cache.items()
    ]

    all_outliers = [o for o in all_outliers if not o.empty]
    if not all_outliers:
        return pd.DataFrame()

    all_outliers_df = pd.concat(all_outliers, ignore_index=True)
    all_outliers_df = apply_fdr(all_outliers_df)

    fdr_pass     = all_outliers_df["p_adj"].fillna(1)       <  FDR_THRESHOLD
    cpg_pass     = all_outliers_df["n_cpg_sites"].fillna(0) >= MIN_CPG_SITES
    iqr_dev_pass = all_outliers_df["iqr_dev"].fillna(0)     >= IQR_DEV_THRESHOLD
    iqr_var_pass = all_outliers_df["iqr"].fillna(1)         <= IQR_MAX_THRESHOLD
    dev_pass     = build_deviation_mask(all_outliers_df, abs_threshold)

    return all_outliers_df[
        fdr_pass & cpg_pass & dev_pass & iqr_dev_pass & iqr_var_pass
    ].copy()


def collect_all_tested_windows(sample_cache, background_df):
    """
    Collect all windows with coverage across every sample.

    Used by `summarise_to_islands` to compute the correct denominator
    for the outlier fraction (total windows tested, not just outlier windows).

    Returns
    -------
    DataFrame with columns: CpG_island, sample.
    """
    frames = []
    for identifier, (_, sample_stats) in sample_cache.items():
        stats_indexed = sample_stats.reindex(background_df.index).fillna(0)
        covered = stats_indexed.index[stats_indexed["total_reads"] > 0]
        if len(covered):
            frames.append(pd.DataFrame({
                "CpG_island": covered,
                "sample":     identifier,
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise_to_islands(outlier_windows_df, windows_df, all_windows_tested_df):
    """
    Aggregate window-level outlier results back to CpG island level.

    For each sample/island combination, reports how many windows were tested,
    how many were significant outliers, and summary statistics for the most
    deviant outlier window.

    Islands are retained only if they satisfy both:
    * at least MIN_OUTLIER_WINDOWS windows are outliers, and
    * the outlier fraction exceeds MIN_OUTLIER_FRACTION.

    Parameters
    ----------
    outlier_windows_df    : window-level outlier DataFrame from `test_sample_meth`
    windows_df            : full windows DataFrame with parent_island column
    all_windows_tested_df : all covered windows across all samples (from
                            `collect_all_tested_windows`)

    Returns
    -------
    DataFrame with one row per sample/island combination, sorted by
    outlier_fraction descending.
    """
    if outlier_windows_df.empty:
        return pd.DataFrame()

    window_to_island = windows_df.set_index("location")["parent_island"]

    outlier_windows_df = outlier_windows_df.copy()
    outlier_windows_df["parent_island"] = (
        outlier_windows_df["CpG_island"].map(window_to_island)
    )

    all_windows_tested_df = all_windows_tested_df.copy()
    all_windows_tested_df["parent_island"] = (
        all_windows_tested_df["CpG_island"].map(window_to_island)
    )

    windows_tested_counts = (
        all_windows_tested_df
        .groupby(["sample", "parent_island"])
        .size()
        .rename("n_windows_tested")
    )

    results = []
    for (sample, island), group in outlier_windows_df.groupby(["sample", "parent_island"]):
        n_outliers = len(group)
        n_total    = (
            windows_tested_counts[(sample, island)]
            if (sample, island) in windows_tested_counts.index
            else n_outliers
        )
        outlier_fraction = n_outliers / n_total if n_total > 0 else 0
        best = group.loc[group["abs_deviation"].idxmax()]

        results.append({
            "CpG_island":          island,
            "sample":              sample,
            "n_windows_tested":    n_total,
            "n_windows_outlier":   n_outliers,
            "outlier_fraction":    round(outlier_fraction, 3),
            "max_abs_deviation":   round(best["abs_deviation"], 4),
            "best_window":         best["CpG_island"],
            "best_window_meth":    round(best["sample_meth"], 4),
            "best_window_mean":    round(best["mean"], 4),
            "best_window_padj":    best["p_adj"],
            "direction":           "hyper" if best["deviation"] > 0 else "hypo",
            "best_window_iqr":     round(best["iqr"], 4),
            "best_window_iqr_dev": round(best["iqr_dev"], 2),
        })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    keep = (
        (results_df["n_windows_outlier"] >= MIN_OUTLIER_WINDOWS) &
        (results_df["outlier_fraction"]  >= MIN_OUTLIER_FRACTION)
    )
    return results_df[keep].sort_values("outlier_fraction", ascending=False)


def combine_results(result_list):
    """
    Concatenate a list of result DataFrames, ignoring empty ones.

    Used to merge outlier results across autosomal, X, and Y analyses.

    Returns an empty DataFrame if all inputs are empty.
    """
    non_empty = [r for r in result_list if not r.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def write_out(allele_results, combined_results):
    """Write allele-level and combined outlier results to tab-separated files."""
    allele_results.to_csv(
        "haplotype_results.txt",   sep="\t", index=False
    )
    combined_results.to_csv(
        "combined_results.txt", sep="\t", index=False
    )


def main():
    # --- Load regions and sex annotations ---
    cpg_df = get_cpg_islands(candidate_file)
    print(f"CpG islands loaded: {len(cpg_df)}")

    males, females = get_sample_sex(sex_file)

    autosomal_cpg, x_cpg, y_cpg = split_cpg_by_chrom(cpg_df)
    print(
        f"Autosomal regions: {len(autosomal_cpg)}, "
        f"X regions: {len(x_cpg)}, "
        f"Y regions: {len(y_cpg)}"
    )

    # --- Build sliding windows ---
    print("\nBuilding sliding windows...")
    windows_auto = build_windows_df(autosomal_cpg)
    windows_x    = build_windows_df(x_cpg)
    windows_y    = build_windows_df(y_cpg)

    # --- Allele-level processing (autosomes: all samples) ---
    print("\nProcessing autosomes (allele-level)...")
    bg_auto, cache_auto, raw_auto = process_files_separately(directory, windows_auto)

    # --- Diploid processing (autosomes) ---
    print("\nProcessing autosomes (combined)...")
    bg_auto_comb, cache_auto_comb = process_files_together(
        directory, windows_auto, raw_auto
    )

    # --- X chromosome: females and males processed separately ---
    print("\nProcessing X chromosome (allele-level, females)...")
    bg_x_f, cache_x_f, raw_x_f = process_files_separately(
        directory, windows_x, samples=females
    )
    print("\nProcessing X chromosome (allele-level, males)...")
    bg_x_m, cache_x_m, raw_x_m = process_files_separately(
        directory, windows_x, samples=males
    )
    print("\nProcessing X chromosome (combined, females)...")
    bg_x_f_comb, cache_x_f_comb = process_files_together(
        directory, windows_x, raw_x_f, samples=females
    )
    print("\nProcessing X chromosome (combined, males)...")
    bg_x_m_comb, cache_x_m_comb = process_files_together(
        directory, windows_x, raw_x_m, samples=males
    )

    # --- Y chromosome: males only ---
    print("\nProcessing Y chromosome (allele-level)...")
    bg_y, cache_y, raw_y = process_files_separately(
        directory, windows_y, samples=males
    )
    print("\nProcessing Y chromosome (combined)...")
    bg_y_comb, cache_y_comb = process_files_together(
        directory, windows_y, raw_y, samples=males
    )

    # --- Window-level outlier testing ---
    print("\nTesting outliers...")
    allele_windows = combine_results([
        test_sample_meth(cache_auto, bg_auto, abs_threshold=ABS_DEV_THRESHOLD_ALLELE),
        test_sample_meth(cache_x_f,  bg_x_f,  abs_threshold=ABS_DEV_THRESHOLD_ALLELE),
        test_sample_meth(cache_x_m,  bg_x_m,  abs_threshold=ABS_DEV_THRESHOLD_ALLELE),
        test_sample_meth(cache_y,    bg_y,    abs_threshold=ABS_DEV_THRESHOLD_ALLELE),
    ])
    combined_windows = combine_results([
        test_sample_meth(cache_auto_comb,  bg_auto_comb,  abs_threshold=ABS_DEV_THRESHOLD_COMBINED),
        test_sample_meth(cache_x_f_comb,   bg_x_f_comb,   abs_threshold=ABS_DEV_THRESHOLD_COMBINED),
        test_sample_meth(cache_x_m_comb,   bg_x_m_comb,   abs_threshold=ABS_DEV_THRESHOLD_COMBINED),
        test_sample_meth(cache_y_comb,     bg_y_comb,     abs_threshold=ABS_DEV_THRESHOLD_COMBINED),
    ])

    # --- Collect tested-window denominators for outlier fraction ---
    all_windows_df = pd.concat([windows_auto, windows_x, windows_y], ignore_index=True)

    all_caches_allele   = {**cache_auto,      **cache_x_f,      **cache_x_m,      **cache_y}
    all_caches_combined = {**cache_auto_comb, **cache_x_f_comb, **cache_x_m_comb, **cache_y_comb}
    all_bgs_allele      = pd.concat([bg_auto,      bg_x_f,      bg_x_m,      bg_y])
    all_bgs_combined    = pd.concat([bg_auto_comb, bg_x_f_comb, bg_x_m_comb, bg_y_comb])

    tested_allele   = collect_all_tested_windows(all_caches_allele,   all_bgs_allele)
    tested_combined = collect_all_tested_windows(all_caches_combined, all_bgs_combined)

    # --- Summarise window-level results to island level ---
    print("\nSummarising to island level...")
    allele_results   = summarise_to_islands(allele_windows,   all_windows_df, tested_allele)
    combined_results = summarise_to_islands(combined_windows, all_windows_df, tested_combined)

    print(f"Allele-level island outliers:   {len(allele_results)}")
    print(f"Combined-level island outliers: {len(combined_results)}")

    write_out(allele_results, combined_results)


if __name__ == "__main__":
    main()
