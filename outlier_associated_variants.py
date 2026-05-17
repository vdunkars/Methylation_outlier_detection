import pandas as pd
from pysam import VariantFile
import csv

def get_region_variants(region_iterator):
    """
    Extract variant-level genotype information from a VCF region iterator.

    Iterates over VCF records in a genomic region and builds a dictionary
    keyed by chromosomal position.

    For each variant:
    - stores reference and alternate alleles
    - collects non-reference genotypes per sample
    - skips missing or homozygous reference genotypes

    Parameters
    ----------
    region_iterator : pysam.VariantFile.fetch iterator
        Iterator over VCF records for a genomic region.

    Returns
    -------
    dict
        Dictionary keyed by "chrom:pos", where each entry contains:
        - ref: reference allele
        - alt: alternate allele(s)
        - genotypes: dict mapping sample → genotype tuple
    """
    region_dict = {}
 
    for record in region_iterator:
        key = record.chrom + ":" + str(record.pos)
        variant_dict = {}
        variant_dict["ref"] = record.ref
        variant_dict["alt"] = record.alts
        genotypes = {}

        for sample_name, sample_data in record.samples.items():
            gt = sample_data.get("GT")
            if gt is None:
                continue
            if None in gt:
                continue

            # skip homozygous reference
            if tuple(sorted(gt)) == (0, 0):
                continue

            genotype_tuple = tuple(sorted(gt))
            genotypes[sample_name] = genotype_tuple

        if len(genotypes) > 0:
            variant_dict["genotypes"] = genotypes
            region_dict[key] = variant_dict

    return region_dict


def process_cpg_region(cpg_region, group, vcf_file, window):
    """
    Extract variants around a CpG region and identify those shared by all outlier samples.

    This function:
    - parses CpG genomic coordinates
    - identifies outlier samples for the region
    - fetches nearby variants from a VCF file
    - filters variants carried by exactly all outlier samples

    Parameters
    ----------
    cpg_region : str
        Genomic region in format "chr:start-end".
    group : pandas.DataFrame
        Subset of results corresponding to one CpG region.
    vcf_file : pysam.VariantFile
        Open VCF file object.
    window : int
        Flanking distance added to region boundaries.

    Returns
    -------
    tuple
        (shared_variants, outlier_samples)
        - shared_variants: dict of filtered variants
        - outlier_samples: list of sample IDs in region
    """
    chrom, coords = cpg_region.split(":")
    start_str, end_str = coords.split("-")
    start = int(start_str)
    end = int(end_str)
    outlier_samples = list(set(group["base_sample"]))

    if len(outlier_samples) == 0:
        return None, None
    region_iterator = vcf_file.fetch(
        contig=chrom,
        start=max(0, start - window),
        stop=end + window
    )

    region_variants = get_region_variants(region_iterator)
    shared_variants = {}

    for pos, var in region_variants.items():
        carriers = list(var["genotypes"].keys())
        if set(carriers) == set(outlier_samples):
            shared_variants[pos] = var

    return shared_variants, outlier_samples



def build_output_rows(cpg_region, group, shared_variants, outlier_samples):
    """
    Build final output rows for a CpG region using shared variants.

    For each outlier sample:
    - attaches CpG region metadata
    - summarizes shared variants
    - formats positions, reference, and alternate alleles as strings

    Parameters
    ----------
    cpg_region : str
        CpG island identifier (chrom:start-end).
    group : pandas.DataFrame
        Data for one CpG region.
    shared_variants : dict
        Variants shared across all outlier samples.
    outlier_samples : list of str
        Samples associated with this CpG region.

    Returns
    -------
    list of dict
        Each dict represents one sample-region summary row.
    """
    output_rows = []
    variants_list = list(shared_variants.values())

    if len(variants_list) == 0:
        return output_rows
    for sample in outlier_samples:
        full_name = group[group["base_sample"] == sample]["sample"].values[0]
        direction = group[group["base_sample"] == sample]["direction"].values[0]
        positions_list = []
        refs_list = []
        alts_list = []

        for v in variants_list:
            positions_list.append(str(v["pos"]) if "pos" in v else "")
            refs_list.append(v["ref"])
            alt = v["alt"]
            if isinstance(alt, (tuple, list)):
                alt_str = ",".join(alt)
            else:
                alt_str = str(alt)

            alts_list.append(alt_str)

        output_rows.append({
            "sample": full_name,
            "region": cpg_region,
            "direction": direction,
            "n_variants": len(variants_list),
            "positions": ",".join(positions_list),
            "refs": ",".join(refs_list),
            "alts": ",".join(alts_list)
        })

    return output_rows



def process_outliers(results_file, vcf_file, window=1000):
    """
    Main pipeline for linking methylation outlier regions to nearby shared genetic variants.

    This function:
    - loads CpG outlier results
    - classifies samples as hyper- or hypo-methylated
    - groups results by CpG island
    - identifies variants shared by all outlier samples in each region
    - builds structured output rows for downstream analysis

    Parameters
    ----------
    results_file : str
        Path to tab-delimited file containing CpG outlier results.
    vcf_file : pysam.VariantFile
        Open VCF file containing genotype data.
    window : int, optional
        Flanking region size around CpG islands (default is 1000 bp).

    Returns
    -------
    list of dict
        Final structured output rows for all CpG regions.
    """
    
    results_df = pd.read_csv(results_file, sep="\t")

    if "max_abs_deviation" not in results_df.columns:
        raise ValueError("missing max_abs_deviation column")
    directions = []

    for x in results_df["max_abs_deviation"].astype(float):
        if x > 0:
            directions.append("hyper")
        else:
            directions.append("hypo")

    results_df["direction"] = directions
    base_samples = []

    for s in results_df["sample"]:
        base_samples.append(s.split("_")[0])
    results_df["base_sample"] = base_samples
    output_rows = []

    for cpg_region, group in results_df.groupby("CpG_island"):
        shared_variants, outlier_samples = process_cpg_region(
            cpg_region,
            group,
            vcf_file,
            window
        )
        if shared_variants is None:
            continue
        rows = build_output_rows(
            cpg_region,
            group,
            shared_variants,
            outlier_samples
        )
        output_rows.extend(rows)
    return output_rows



def write_output(rows, filename):
    """
    Write processed CpG-variant association results to a TSV file.

    Each row contains:
    - sample identifier
    - CpG region
    - methylation direction (hyper/hypo)
    - number of shared variants
    - variant positions, reference and alternate alleles

    Parameters
    ----------
    rows : list of dict
        Output rows from process_outliers.
    filename : str
        Output file path.

    Returns
    -------
    None
    """
    if len(rows) == 0:
        print("No results found.")
        return
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "Sample",
            "Region",
            "Direction",
            "N_variants",
            "Positions",
            "Ref",
            "Alt"
        ])
        for r in rows:
            writer.writerow([
                r["sample"],
                r["region"],
                r["direction"],
                r["n_variants"],
                r["positions"],
                r["refs"],
                r["alts"]
            ])


def main():
    results_file = ""
    vcf_filename = ""
    output_file = ""
    vcf_file = VariantFile(vcf_filename)
    rows = process_outliers(results_file, vcf_file, window=1000)
    write_output(rows, output_file)
    vcf_file.close()
    print("Done. Rows written:", len(rows))


if __name__ == "__main__":
    main()