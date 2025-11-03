import glob
import json
import os
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd


class CaseMethodReslts:
    def __init__(self, case, nm):
        self.case = case
        self.nm = nm
        self.metrics = defaultdict(list)

    def add_method_rep(self, method_results):
        for met in method_results["metrics"]:
            self.metrics[met] += [method_results["metrics"][met]]


class CaseReslts:
    def __init__(self, case):
        self.case = case
        self.method_results = {}

    def add_reps(self, all_results):
        for rep in range(len(all_results)):
            for method_nm in all_results[rep]:
                one_result = all_results[rep][method_nm]
                self._add_rep(one_result)

    def _add_rep(self, one_result):
        nm = one_result["mth"]
        if nm not in self.method_results:
            self.method_results[nm] = CaseMethodReslts(self.case, nm)
        self.method_results[nm].add_method_rep(one_result)

    def write_case(self, params, exp, options):
        options.logger.info("")
        options.logger.info(f"***RESULTS***  Case: {self.case}")
        table = self.method_results
        base_attribute_idf = options.get_base_attribute_idf()
        long_path_pre = os.path.join(options.out_dir, os.path.join(
            str(options.exp_type) + "_" + '_'.join([f"{e}_{exp[e]}" for e in exp])), base_attribute_idf)
        path_pre = os.path.join(options.out_dir, base_attribute_idf)

        path = os.path.join(path_pre, os.path.join("tikzfiles", "all/"))
        os.makedirs(path, exist_ok=True)

        # results of each run
        methods = table.keys()
        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            with open(fl, 'w') as write_file:
                write_file.write('X')
                for met in table[mth].metrics.keys():
                    write_file.write(f'\t{str(mth)}_{met}')
                for r in range(options.reps):
                    write_file.write(f'\n{r}')
                    for met in table[mth].metrics.keys():
                        if len(table[mth].metrics[met]) > r:
                            write_file.write(f'\t{table[mth].metrics[met][r]} ')
                        else:
                            write_file.write('\t-1')

        # Print averages of all runs for this case
        path = os.path.join(path_pre, "tikzfiles/avg/")
        os.makedirs(path, exist_ok=True)

        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            with open(fl, 'w') as write_file:
                write_file.write('X')
                for met in table[mth].metrics.keys():
                    write_file.write(
                        f'\t{str(mth)}_{met}_mn\t{str(mth)}_{met}_md\t{str(mth)}_{met}_var\t{str(mth)}_{met}_std')
                write_file.write('\n0')

                for met in table[mth].metrics.keys():
                    vals_raw = table[mth].metrics[met]
                    relevant_entries = [_as_missing(x) for x in vals_raw if x != -1]
                    # choose weights if metric benefits from it
                    weights = _weights_for_metric(table, mth, met)

                    if len(relevant_entries) >= 1:
                        try:
                            mn = safe_mean(vals_raw, weights=weights)
                            md = safe_median(vals_raw)
                            var = safe_var(vals_raw)
                            std = safe_std(vals_raw)

                            # Fallbacks as -1 when NaN to keep your file format stable
                            def _fmt(v):
                                return (str(v) if not (isinstance(v, float) and np.isnan(v)) else '-1')

                            write_file.write(f'\t{_fmt(mn)}\t{_fmt(md)}\t{_fmt(var)}\t{_fmt(std)}')
                        except Exception:
                            write_file.write('\t-1\t-1\t-1\t-1')
                    else:
                        write_file.write('\t-1\t-1\t-1\t-1')

        # Print only means
        path = os.path.join(path_pre, "tikzfiles/avgmn/")
        os.makedirs(path, exist_ok=True)

        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            with open(fl, 'w') as write_file:
                write_file.write('X')
                for met in table[mth].metrics.keys():
                    write_file.write(f'\t{met}')
                write_file.write('\n0')

                for met in table[mth].metrics.keys():
                    vals_raw = table[mth].metrics[met]
                    weights = _weights_for_metric(table, mth, met)
                    mn = safe_mean(vals_raw, weights=weights)
                    if isinstance(mn, float) and np.isnan(mn):
                        write_file.write('\t-1')
                    else:
                        write_file.write(f'\t{np.round(mn, 2)}')

            # pretty print the last file written
            import csv
            from tabulate import tabulate
            options.logger.info(f"\tMethod: {mth}")
            with open(fl) as csv_file:
                reader = csv.reader(csv_file, delimiter='\t')
                rows = [row for row in reader]
                options.logger.info(tabulate(rows, tablefmt="pretty"))


    def write_case_previous(self, params, exp, options):
        options.logger.info("")
        options.logger.info(f"***RESULTS***  Case: {self.case}")
        table = self.method_results
        base_attribute_idf = options.get_base_attribute_idf()
        long_path_pre = os.path.join(options.out_dir, os.path.join(str(options.exp_type) + "_" + '_'.join([f"{e}_{exp[e]}" for e in exp])),  base_attribute_idf)
        path_pre = os.path.join(options.out_dir, base_attribute_idf)

        path = os.path.join(path_pre, os.path.join("tikzfiles", "all/"))
        os.makedirs(path, exist_ok=True)

        # results of each run
        methods = table.keys()
        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            write_file = open(fl, 'w')
            write_file.write(f'X')

            for met in table[mth].metrics.keys():
                write_file.write(f'\t{str(mth)}_{met}')
            for r in range(options.reps):
                write_file.write(f'\n{r}')
                for met in table[mth].metrics.keys():
                    if len(table[mth].metrics[met]) > r:
                        write_file.write(f'\t{table[mth].metrics[met][r]} ')
                    else:
                        write_file.write(f'\t{-1}')
            write_file.close()

        # Print averages of all runs for this case
        path = os.path.join(path_pre, "tikzfiles/avg/")

        if not os.path.exists(path):
            os.makedirs(path)

        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            write_file = open(fl, 'w')
            write_file.write(f'X')

            for met in table[mth].metrics.keys():
                write_file.write(
                    f'\t{str(mth)}_{met}_mn\t{str(mth)}_{met}_md\t{str(mth)}_{met}_var\t{str(mth)}_{met}_std')
            write_file.write(f'\n0')
            for met in table[mth].metrics.keys():
                relevant_entries = [x for x in table[mth].metrics[met] if x != -1]  # -1 is placeholder for NaN
                if len(relevant_entries) > 2:
                    try:
                        write_file.write(
                            f'\t{statistics.mean(relevant_entries)}'
                            f'\t{statistics.median(relevant_entries)}'
                            f'\t{statistics.variance(relevant_entries)}'
                            f'\t{statistics.stdev(relevant_entries)}')
                    except:
                        write_file.write(f'\t-1\t-1\t-1\t-1')
                elif len(table[mth].metrics[met]) == 1:
                    write_file.write(f'\t{relevant_entries[0]}\t0\t0\t0')
                else:
                    write_file.write(f'\t{-1}\t0\t0\t0')
            write_file.close()

        # Print only means
        path = os.path.join(path_pre, "tikzfiles/avgmn/")

        if not os.path.exists(path):
            os.makedirs(path)

        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            write_file = open(fl, 'w')
            write_file.write(f'X')

            for met in table[mth].metrics.keys():
                write_file.write(
                    f'\t{met}')
            write_file.write(f'\n0')
            for met in table[mth].metrics.keys():
                relevant_entries = [x for x in table[mth].metrics[met] if x != -1]  # -1 is placeholder for NaN
                if len(relevant_entries) > 2:
                    try:
                        write_file.write(
                            f'\t{np.round(statistics.mean(relevant_entries), 2)}')
                    except:
                        write_file.write(f'\t-1')
                elif len(relevant_entries) == 1:
                    write_file.write(f'\t{relevant_entries[0]}')
                else:
                    write_file.write(f'\t{-1}')
            write_file.close()

            import csv
            from tabulate import tabulate

            options.logger.info(f"\tMethod: {mth}")

            with open(fl) as csv_file:
                reader = csv.reader(csv_file, delimiter='\t')
                rows = [row for row in reader]
                options.logger.info(tabulate(rows, tablefmt="pretty"))

"""
def write_cases(options, exp, relevant_attribute, read_dir=None):
    # base_attribute_idf identifies a unique parameter config (which attributes we keep fixed when plotting another)
    base_attributes = options.fixed
    base_attribute_idf = options.get_base_attribute_idf()  # '_'.join([f'{ky}_{vl}' for ky, vl in base_attributes.items()])

    fixed_attributes = {ky: vl for (ky, vl) in base_attributes.items() if ky != relevant_attribute}
    fixed_attribute_idf = '_'.join([f'{ky}_{vl}' for ky, vl in fixed_attributes.items()])
    experiment_idf = "" # str(options.exp_type) + "_" +   '_'.join([f"{e}_{exp[e]}" for e in exp])

    in_pre = os.path.join(os.path.join(options.out_dir, experiment_idf), base_attribute_idf) if read_dir is None else \
        os.path.join(os.path.join(read_dir, experiment_idf), base_attribute_idf)
    out_pre = os.path.join(os.path.join(options.out_dir, experiment_idf), base_attribute_idf)
    in_path = os.path.join(os.path.join(in_pre, "tikzfiles"), "avg")

    out_path = os.path.join(os.path.join(out_pre, "tikzfiles"), "change")
    os.makedirs(out_path, exist_ok=True)

    # info on used attributes for reference
    info_pth = os.path.join(out_path, f"info")
    info_fl = os.path.join(info_pth, f"change_{relevant_attribute}")
    os.makedirs(info_pth, exist_ok=True)
    with open(f'{info_fl}.json', 'w') as fp:
        json.dump(fixed_attributes, fp)
    with open(f'{os.path.join(info_pth, f"base_config")}.json', 'w') as fp:
        json.dump(base_attributes, fp)

    fd_metrics = {}
    fd_attrvals = {}
    for fl in glob.glob(os.path.join(in_path, "*.tsv")):
        mthd, attribute_value, contains_base_attributes = None, None, True
        fl = fl.replace('\\', '/')
        suff = fl.split("/")[len(fl.split("/")) - 1].split('.tsv')[0]
        parts = suff.split('_')
        # check all base attrs covered
        for ip, p in enumerate(parts):
            if p in fixed_attributes:
                contains_base_attributes = contains_base_attributes and parts[ip + 1] == str(fixed_attributes[p])
        if not contains_base_attributes:
            continue

        # extract method and results
        for ip, p in enumerate(parts):
            if p != 'm': continue
            mthd = parts[ip + 1]
            if mthd not in fd_metrics: fd_metrics[mthd] = {}
            if mthd not in fd_attrvals: fd_attrvals[mthd] = []

        for ip, p in enumerate(parts):
            if p != relevant_attribute: continue
            attribute_value = parts[ip + 1]

            if attribute_value not in fd_attrvals[mthd]: fd_attrvals[mthd].append(attribute_value)
        assert attribute_value is not None and mthd is not None
        tb = pd.read_csv(fl, sep='\t')
        for metr in tb.columns:
            if metr == 'X': continue
            if metr not in fd_metrics[mthd]: fd_metrics[mthd][metr] = {}
            fd_metrics[mthd][metr][attribute_value] = tb[metr].iloc[0]

    for mthd in fd_attrvals:
        fd_attrvals[mthd] = sorted(fd_attrvals[mthd], key=float)

    # For each method, create the following file
    # cols: method_metric1_mn, method_metric1_var, method_metric1_std ... method_metricN_std
    # rows: value1(relevant_attribute) .... valueN (relevant_attribute)

    def check_all_present(mth, mt):
        return np.all(
            [attr_v in fd_metrics[mth][mt] and (fd_metrics[mth][mt][attr_v] is not None) for attr_v in
             fd_attrvals[mth]])

    for mthd in fd_metrics:
        # informative path:
        # out_fl = os.path.join(out_path, f"{base_attribute_idf}_{mthd}.tsv")
        # short path:
        out_fl = os.path.join(out_path, f"change_{relevant_attribute}_m_{mthd}.tsv")

        write_file = open(out_fl, 'w')
        write_file.write(f'{relevant_attribute}')

        # 1st row, metric names
        for met in fd_metrics[mthd].keys():
            if not check_all_present(mthd, met): continue

            # column name is metric_statistic
            parts = met.split('_')
            nm_metric, nm_statistic = parts[-2], parts[-1]
            assert nm_statistic in ['mn', 'md', 'var', 'std']
            # write_file.write(f'\t{str(met)}') #if method name should be in front
            write_file.write(f'\t{nm_metric}_{nm_statistic}')

        # other rows, each value
        for attr_val in fd_attrvals[mthd]:
            write_file.write(f'\n{attr_val}')
            for met in fd_metrics[mthd].keys():
                if not check_all_present(mthd, met): continue
                assert attr_val in fd_metrics[mthd][met] and (fd_metrics[mthd][met][attr_val] is not None)
                write_file.write(f'\t{fd_metrics[mthd][met][attr_val]:.5f}')

        write_file.close()

        # in addition log resulting file content
        import csv
        from tabulate import tabulate
        with open(out_fl) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            rows = [row for row in reader]
            options.logger.info(
                f"Method: {mthd}, Attribute: {relevant_attribute}, base: {base_attributes} File: {out_fl}")
            options.logger.info(tabulate(rows, tablefmt="pretty"))

"""

import os, glob, json
import numpy as np
import pandas as pd

def _parse_filename_kv(parts):
    """
    Parse tokens like ["case","N","5","S","100","m","MyMethod", ...] or ["N_5","S_100","m","MyMethod"].
    Returns (kv, method) where kv is dict of key->value (strings), and method is str or None.
    """
    kv = {}
    method = None

    # First pass: tokens that look like key_val
    for tok in parts:
        if "_" in tok:
            k, v = tok.split("_", 1)
            if k and v:  # keep first occurrence; later explicit 'm' parsing may overwrite
                kv.setdefault(k, v)

    # Second pass: explicit pairwise tokens ... k v ...
    i = 0
    while i < len(parts) - 1:
        k, v = parts[i], parts[i+1]
        if k == "m":
            method = v
            i += 2
            continue
        # Only add if not already found by key_val pattern
        if (k not in kv) and (k not in ("", "m")):
            kv[k] = v
            i += 2
            continue
        i += 1

    # Also try to find method if not set yet
    if method is None:
        for i, tok in enumerate(parts[:-1]):
            if tok == "m":
                method = parts[i+1]
                break

    return kv, method

def _all_base_attrs_present(parsed_kv, fixed_attributes):
    """True if every key in fixed_attributes is present in parsed_kv and value matches (as string)."""
    for k, v in fixed_attributes.items():
        if k not in parsed_kv:
            return False
        if str(parsed_kv[k]) != str(v):
            return False
    return True

def _try_sort_numeric(values):
    """Return (sorted_values, used_numeric) where used_numeric indicates if numeric sort applied."""
    try:
        vals = [float(v) for v in values]
        order = np.argsort(vals)
        return [values[i] for i in order], True
    except Exception:
        return sorted(values), False

def _fmt_val5(x):
    """Format float to 5 decimals; if nan, return -1; pass-through for ints/strings."""
    try:
        xf = float(x)
        if np.isnan(xf):
            return "-1"
        return f"{xf:.5f}"
    except Exception:
        return str(x)

def write_cases(options, exp, relevant_attribute, read_dir=None):
    # base_attribute_idf identifies a unique parameter config
    base_attributes = options.fixed
    base_attribute_idf = options.get_base_attribute_idf()

    fixed_attributes = {ky: vl for (ky, vl) in base_attributes.items() if ky != relevant_attribute}
    experiment_idf = ""

    in_pre = os.path.join(os.path.join(options.out_dir, experiment_idf), base_attribute_idf) if read_dir is None else \
        os.path.join(os.path.join(read_dir, experiment_idf), base_attribute_idf)
    out_pre = os.path.join(os.path.join(options.out_dir, experiment_idf), base_attribute_idf)
    in_path = os.path.join(os.path.join(in_pre, "tikzfiles"), "avg")
    out_path = os.path.join(os.path.join(out_pre, "tikzfiles"), "change")
    os.makedirs(out_path, exist_ok=True)

    # info on used attributes for reference
    info_pth = os.path.join(out_path, "info")
    info_fl = os.path.join(info_pth, f"change_{relevant_attribute}")
    os.makedirs(info_pth, exist_ok=True)
    with open(f'{info_fl}.json', 'w') as fp:
        json.dump(fixed_attributes, fp)
    with open(f'{os.path.join(info_pth, "base_config")}.json', 'w') as fp:
        json.dump(base_attributes, fp)

    fd_metrics = {}   # method -> metric_name -> { attr_val: value }
    fd_attrvals = {}  # method -> list of attr values (strings)

    for fl in glob.glob(os.path.join(in_path, "*.tsv")):
        fl_norm = fl.replace('\\', '/')
        suffix = os.path.basename(fl_norm).split('.tsv')[0]
        parts = suffix.split('_')

        parsed_kv, mthd = _parse_filename_kv(parts)
        if mthd is None:
            # skip files without a method token
            continue
        if not _all_base_attrs_present(parsed_kv, fixed_attributes):
            continue
        if relevant_attribute not in parsed_kv:
            # this file doesn’t vary relevant_attribute; skip
            continue

        attr_val = parsed_kv[relevant_attribute]

        if mthd not in fd_metrics:
            fd_metrics[mthd] = {}
        if mthd not in fd_attrvals:
            fd_attrvals[mthd] = []
        if attr_val not in fd_attrvals[mthd]:
            fd_attrvals[mthd].append(attr_val)

        tb = pd.read_csv(fl_norm, sep='\t')
        # Each avg file has a single data row (row index '0')
        if tb.shape[0] == 0:
            continue

        for metr in tb.columns:
            if metr == 'X':
                continue
            if metr not in fd_metrics[mthd]:
                fd_metrics[mthd][metr] = {}
            val = tb[metr].iloc[0]
            # Keep NaNs as NaN; downstream _fmt_val5 will map to -1
            try:
                v = float(val)
            except Exception:
                v = np.nan
            fd_metrics[mthd][metr][attr_val] = v

    # Sort attribute values per method (numeric if possible)
    for mthd in fd_attrvals:
        fd_attrvals[mthd], _ = _try_sort_numeric(fd_attrvals[mthd])

    # Helper to check completeness for a metric across all attr values for a method
    def check_all_present(mth, mt):
        return all(attr_v in fd_metrics[mth][mt] for attr_v in fd_attrvals[mth])

    # For each method, create the “change” file
    for mthd in fd_metrics:
        out_fl = os.path.join(out_path, f"change_{relevant_attribute}_m_{mthd}.tsv")
        with open(out_fl, 'w') as write_file:
            write_file.write(f'{relevant_attribute}')
            # Header: metric_statistic (drop method prefix if present)
            # Deterministic order
            metric_names = sorted(fd_metrics[mthd].keys())
            header_metrics = []
            for met in metric_names:
                if not check_all_present(mthd, met):
                    continue
                parts = met.split('_')
                # Expect end tokens *_mn, *_md, *_var, *_std; be robust if names contain underscores
                if len(parts) >= 2 and parts[-1] in ('mn', 'md', 'var', 'std'):
                    nm_stat = parts[-1]
                    nm_metric = "_".join(parts[:-1])
                    header_metrics.append((met, f"{nm_metric}_{nm_stat}"))
                else:
                    header_metrics.append((met, met))
            for _, hdr in header_metrics:
                write_file.write(f'\t{hdr}')

            # Rows per attribute value
            for attr_val in fd_attrvals[mthd]:
                write_file.write(f'\n{attr_val}')
                for met, _hdr in header_metrics:
                    v = fd_metrics[mthd][met].get(attr_val, np.nan)
                    write_file.write(f'\t{_fmt_val5(v)}')

        # pretty log
        import csv
        from tabulate import tabulate
        with open(out_fl) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            rows = [row for row in reader]
            options.logger.info(
                f"Method: {mthd}, Attribute: {relevant_attribute}, base: {base_attributes} File: {out_fl}")
            options.logger.info(tabulate(rows, tablefmt="pretty"))

import os, statistics

def _as_missing(x):
    # interpret placeholder -1 as missing
    return np.nan if (x == -1) else x

def safe_mean(vals, weights=None):
    vals = np.asarray([_as_missing(v) for v in vals], float)
    if weights is None:
        vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)) if vals.size else float('nan')
    w = np.asarray(weights, float)
    mask = (~np.isnan(vals)) & (w > 0)
    if not mask.any():
        return float('nan')
    return float(np.sum(vals[mask] * w[mask]) / np.sum(w[mask]))

def safe_median(vals):
    vals = [ _as_missing(v) for v in vals ]
    vals = [ v for v in vals if not (isinstance(v,float) and np.isnan(v)) ]
    return float(statistics.median(vals)) if len(vals) else float('nan')

def safe_var(vals):
    vals = [ _as_missing(v) for v in vals ]
    vals = [ v for v in vals if not (isinstance(v,float) and np.isnan(v)) ]
    return float(statistics.variance(vals)) if len(vals) >= 2 else float('nan')

def safe_std(vals):
    vals = [ _as_missing(v) for v in vals ]
    vals = [ v for v in vals if not (isinstance(v,float) and np.isnan(v)) ]
    return float(statistics.stdev(vals)) if len(vals) >= 2 else float('nan')

def _weights_for_metric(table, mth, met):
    runs = len(table[mth].metrics[met])
    if met.startswith("micro_"):
        # weight by total_pairs if present
        if "total_pairs" in table[mth].metrics:
            w = table[mth].metrics["total_pairs"]
            return w if len(w) == runs else None
    if met.startswith("macro_variant_"):
        if "n_variant_nodes" in table[mth].metrics:
            w = table[mth].metrics["n_variant_nodes"]
            return w if len(w) == runs else None
    if met.startswith("macro_invariant_"):
        if "n_invariant_nodes" in table[mth].metrics:
            w = table[mth].metrics["n_invariant_nodes"]
            return w if len(w) == runs else None
    return None