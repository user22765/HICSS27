import os
import json
import numpy as np
import pandas as pd
import jellyfish
import matplotlib.pyplot as plt


CONFIG = ["3-tuple", "4-tuple", "5-tuple"]


def _validate_split(split):
    """
    Validate dataset split.
    """
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got: {split!r}")


def _safe_micro_prf(correct, predicted, gold):
    """
    Safely compute micro precision, recall, and F1.
    """
    precision = correct / predicted if predicted > 0 else 0
    recall = correct / gold if gold > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )
    return precision, recall, f1


def _load_test_pubmed_lookup(path2gt):
    """
    For the test split, PubMed label filenames do not directly match prediction filenames.

    This loads:

        <path2gt>/filtered_pubmed_v2.json

    and creates a mapping:

        doi_url -> paper_name_id

    where:

        paper_name_id = downloaded_pdf.replace(".pdf", "")
    """
    path = os.path.join(path2gt, "filtered_pubmed_v2.json")

    with open(path, "r") as f:
        papers = json.load(f)

    doi_to_paper_id = {}

    for paper in papers:
        doi_url = paper.get("doi_url")
        downloaded_pdf = paper.get("downloaded_pdf")

        if not doi_url or not downloaded_pdf:
            continue

        doi_to_paper_id[doi_url] = downloaded_pdf.replace(".pdf", "")

    return doi_to_paper_id


def norm_var(x):
    """
    Normalize variable / construct names before fuzzy comparison.
    """
    if x is None:
        return "none"

    x = str(x).lower().strip()
    x = " ".join(x.split())

    if x == "":
        return "none"

    return x

'''
def _char_ngrams(s, n=2):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}

def _jaccard(a_set, b_set):
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)

def var_match(a, b, threshold=0.9):   # note: lower than JW's 0.90
    a = norm_var(a)
    b = norm_var(b)

    if a == b:
        return True
    if a == "none" or b == "none":
        return False

    return _jaccard(_char_ngrams(a), _char_ngrams(b)) >= threshold

'''    
def var_match(a, b, threshold=0.90):
    """
    Fuzzy variable-name matching using Jaro-Winkler similarity.

    Returns True if the two variable names are similar enough.
    """
    a = norm_var(a)
    b = norm_var(b)

    if a == b:
        return True

    if a == "none" or b == "none":
        return False

    return jellyfish.jaro_winkler_similarity(a, b) >= threshold


def print_low_f1(per_paper_f1, threshold=0.1):
    for conf, scores in per_paper_f1.items():
        low = {fn: f1 for fn, f1 in scores.items() if f1 < threshold}
        print(f"\n{conf}: {len(low)} papers with F1 < {threshold}")
        for fn, f1 in sorted(low.items(), key=lambda x: x[1]):
            print(f"  {f1:.3f}  {fn}")


def print_perfect_f1(per_paper_f1, threshold=0.99999):
    for conf, scores in per_paper_f1.items():
        high = {fn: f1 for fn, f1 in scores.items() if f1 > threshold}
        print(f"\n{conf}: {len(high)} papers with F1 < {threshold}")
        for fn, f1 in sorted(high.items(), key=lambda x: x[1]):
            print(f"  {f1:.3f}  {fn}")


def plot_f1_histogram(per_paper_f1, title="Paper-level F1", bins=20):
    """
    per_paper_f1: {eval_conf: {fname: f1}} as returned by eval_disknet / eval_pubmed.
    One subplot per eval config.
    """
    confs = list(per_paper_f1.keys())
    fig, axes = plt.subplots(1, len(confs),
                             figsize=(5 * len(confs), 4), squeeze=False)

    for ax, conf in zip(axes[0], confs):
        scores = list(per_paper_f1[conf].values())
        ax.hist(scores, bins=bins, range=(0, 1), edgecolor="black")
        ax.axvline(np.mean(scores), color="red", linestyle="--",
                   label=f"mean={np.mean(scores):.2f}")
        ax.set_title(f"{conf}  (n={len(scores)})")
        ax.set_xlabel("F1")
        ax.set_ylabel("papers")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return fig


def eval(
    preds_list,
    gts_list,
    human_eval=True,
    mapping_dict=None,
    include_pc=False,
    include_sig=False,
    ignore_imgs=False,
    var_threshold=0.90,
):
    """
    Evaluates predicted relations against ground truth relations.

    Both preds_list and gts_list are assumed to be lists of dictionaries.

    Ground-truth relation fields:
      {
        "Construct from": ...,
        "Construct to": ...,
        "Construct Moderator": ...,
        "Path coefficient": ...,
        "significance": ...
      }

    Prediction relation fields:
      {
        "cause": ...,
        "effect": ...,
        "beta": ...,
        "p": ...,
        "model_id": ...,
        "moderator": ...,
        "mediator": ...
      }

    Returns:
      correct, n_predictions, n_ground_truth, precision, recall, f1, equivalent_vars
    """
    gt_tups = []
    #ignore_imgs=True

    for gt in gts_list:
        if ignore_imgs:
            try:
                if (
                    not gt["in_text"]
                    and not gt["hyp_in_text"]
                    and not gt["in_table"]
                ):
                    continue
            except Exception:
                print("[ERROR] Relation does not contain in_<text/table> fields:", gt)
                continue

        c1 = norm_var(gt["Construct from"])
        c2 = norm_var(gt["Construct to"])
        cm = norm_var(gt["Construct Moderator"])
        si = str(gt["significance"]).replace(" ", "").lower().strip()
        pc = gt["Path coefficient"]

        if mapping_dict and human_eval:
            if c1 in mapping_dict:
                c1 = norm_var(mapping_dict[c1])
            if c2 in mapping_dict:
                c2 = norm_var(mapping_dict[c2])
            if cm and cm in mapping_dict:
                cm = norm_var(mapping_dict[cm])

        if not isinstance(pc, float):
            pc = None
        else:
            pc = round(pc, 2)

        if not include_pc and not include_sig:
            gt_tups.append((c1, c2, cm))
        elif include_pc and not include_sig:
            gt_tups.append((c1, c2, cm, pc))
        elif not include_pc and include_sig:
            gt_tups.append((c1, c2, cm, si))
        elif include_pc and include_sig:
            gt_tups.append((c1, c2, cm, pc, si))

    gt_tups = list(set(gt_tups))

    if len(gt_tups) == 0:
        return None, None, None, None, None, None, {}

    pred_tups = []

    for p in preds_list:
        c1 = norm_var(p.get("cause", ""))
        c2 = norm_var(p.get("effect", ""))
        cm1 = norm_var(p.get("moderator", ""))
        cm2 = norm_var(p.get("mediator", ""))
        cm = cm1

        if cm2 != "none" and cm1 == "none":
            cm = cm2

        sig = str(p.get("p", "")).replace(" ", "").lower().strip()
        pc = p.get("beta", None)

        if not isinstance(pc, float):
            pc = None
        else:
            pc = round(pc, 2)

        if not include_pc and not include_sig:
            pred_tups.append((c1, c2, cm))
        elif include_pc and not include_sig:
            pred_tups.append((c1, c2, cm, pc))
        elif not include_pc and include_sig:
            pred_tups.append((c1, c2, cm, sig))
        elif include_pc and include_sig:
            pred_tups.append((c1, c2, cm, pc, sig))

    pred_tups = list(set(pred_tups))

    correct = 0
    matched_gt = set()
    equivalent_vars = {}

    for pred in pred_tups:
        for i, gt in enumerate(gt_tups):
            if i in matched_gt:
                continue

            if not var_match(pred[0], gt[0], threshold=var_threshold):
                continue

            if not var_match(pred[1], gt[1], threshold=var_threshold):
                continue

            if not var_match(pred[2], gt[2], threshold=var_threshold):
                continue

            # Exact matching for path coefficient and/or significance.
            if len(pred) > 3 and pred[3:] != gt[3:]:
                continue

            correct += 1
            matched_gt.add(i)

            for pred_var, gt_var in zip(pred[:3], gt[:3]):
                if pred_var == "none" and gt_var == "none":
                    continue

                if pred_var not in equivalent_vars:
                    equivalent_vars[pred_var] = set()

                equivalent_vars[pred_var].add(gt_var)

            break

    p_val = correct / len(pred_tups) if len(pred_tups) > 0 else 0
    r_val = correct / len(gt_tups) if len(gt_tups) > 0 else 0

    f1 = (
        2 * p_val * r_val / (p_val + r_val)
        if p_val + r_val > 0
        else 0
    )

    equivalent_vars = {
        k: sorted(list(v))
        for k, v in equivalent_vars.items()
    }

    return correct, len(pred_tups), len(gt_tups), p_val, r_val, f1, equivalent_vars


def eval_disknet(
    path2preds,
    suffix,
    conf=CONFIG,
    split="val",
    mapping_dict=None,
    verbose=True,
    relations_field=True,
    var_match=0.9,
    path2gt=None,
):
    """
    Evaluate DiskNet / Bo8 predictions.

    Parameters:
      path2preds:
        Directory containing prediction files.

      suffix:
        Prediction filename suffix, e.g. "_run1_cleaned.json", "_ensemble.json", ".json".

      conf:
        Evaluation configurations, usually ["3-tuple", "4-tuple", "5-tuple"].

      split:
        Either "val" or "test".

      path2gt:
        Ground-truth base directory. If None, defaults to f"../{split}".

    Expected ground-truth directory:
      <path2gt>/disknet_labels/
    """
    _validate_split(split)

    if path2gt is None:
        path2gt = f"../{split}"

    human_eval = mapping_dict is not None

    out_prec, out_rec, out_f1 = [], [], []
    equivalent_vars_by_file = {}
    per_paper_f1 = {}

    labels_dir = os.path.join(path2gt, "disknet_labels")

    for eval_conf in conf:
        cs, ps, gs = [], [], []
        precs, recs, f1s = [], [], []

        equivalent_vars_by_file[eval_conf] = {}
        per_paper_f1[eval_conf] = {}

        for fname in sorted(os.listdir(labels_dir)):
            if not fname.endswith(".json"):
                continue

            gt_path = os.path.join(labels_dir, fname)

            with open(gt_path, "r") as gt_file:
                gt = json.load(gt_file)

            gt_rels = gt["relations"]

            pred_path = os.path.join(
                path2preds,
                fname.replace(".json", suffix),
            )

            if not os.path.exists(pred_path):
                print(f"[ERROR] Did not find predictions for:", fname)

                precs.append(0)
                recs.append(0)
                f1s.append(0)
                cs.append(0)
                ps.append(0)
                gs.append(len(gt_rels))
                equivalent_vars_by_file[eval_conf][fname] = {}
                per_paper_f1[eval_conf][fname] = 0 
                continue

            with open(pred_path, "r") as predfile:
                preds = json.load(predfile)

            if relations_field:
                preds = preds["relations"]

            pred_rels = preds
            pred_rels = [p for p in pred_rels if not 'mediator' in p.keys() or not p['mediator']]

            if mapping_dict and fname in mapping_dict:
                paper_mapping = mapping_dict[fname]
            else:
                paper_mapping = None

            if eval_conf == "3-tuple":
                nc, np_, ng, p, r, f1, equivalent_vars = eval(
                    pred_rels,
                    gt_rels,
                    human_eval=human_eval,
                    mapping_dict=paper_mapping,
                    include_pc=False,
                    include_sig=False,
                    ignore_imgs=False,
                    var_threshold=var_match,
                )
            elif eval_conf == "4-tuple":
                nc, np_, ng, p, r, f1, equivalent_vars = eval(
                    pred_rels,
                    gt_rels,
                    human_eval=human_eval,
                    mapping_dict=paper_mapping,
                    include_pc=True,
                    include_sig=False,
                    ignore_imgs=False,
                    var_threshold=var_match,
                )
            elif eval_conf == "5-tuple":
                nc, np_, ng, p, r, f1, equivalent_vars = eval(
                    pred_rels,
                    gt_rels,
                    human_eval=human_eval,
                    mapping_dict=paper_mapping,
                    include_pc=True,
                    include_sig=True,
                    ignore_imgs=False,
                    var_threshold=var_match,
                )
            else:
                raise ValueError(f"Unknown eval config: {eval_conf}")

            equivalent_vars_by_file[eval_conf][fname] = equivalent_vars

            if nc is None:
                continue

            precs.append(p)
            recs.append(r)
            f1s.append(f1)
            cs.append(nc)
            ps.append(np_)
            gs.append(ng)
            per_paper_f1[eval_conf][fname] = f1

        prec, rec, micro_f1 = _safe_micro_prf(sum(cs), sum(ps), sum(gs))

        out_prec.append(round(prec, 4))
        out_rec.append(round(rec, 4))
        out_f1.append(round(micro_f1, 4))

        if verbose:
            print(100 * "=")
            print(f"RESULTS for {eval_conf}:")
            print("Precision:", round(prec, 4))
            print("Recall:", round(rec, 4))
            print("F1:", round(micro_f1, 4))

            if len(precs) > 0:
                print("Macro Precision:", round(sum(precs) / len(precs), 4))
                print("Macro Recall:", round(sum(recs) / len(recs), 4))
                print("Macro F1:", round(sum(f1s) / len(f1s), 4))

            print(150 * "=")

    return out_rec, out_prec, out_f1, equivalent_vars_by_file, per_paper_f1


def eval_pubmed(
    path2preds,
    suffix,
    conf=CONFIG,
    split="val",
    mapping_dict=None,
    verbose=True,
    relations_field=True,
    var_match=0.9,
    path2gt=None,
):
    """
    Evaluate PubMed predictions.

    Parameters:
      path2preds:
        Directory containing prediction files.

      suffix:
        Prediction filename suffix, e.g. "_run1_cleaned.json", "_ensemble.json", ".json".

      conf:
        Evaluation configurations, usually ["3-tuple", "4-tuple", "5-tuple"].

      split:
        Either "val" or "test".

      path2gt:
        Ground-truth base directory. If None, defaults to f"../{split}".

    Expected validation structure:
      <path2gt>/pubmed/<paper_name_id>.json

    Expected test structure:
      <path2gt>/pubmed/
      <path2gt>/filtered_pubmed_v2.json

    For test PubMed, prediction filenames are derived from filtered_pubmed_v2.json:
      paper_name_id = paper["downloaded_pdf"].replace(".pdf", "")
    """
    _validate_split(split)

    if path2gt is None:
        path2gt = f"../{split}"

    human_eval = mapping_dict is not None

    out_prec, out_rec, out_f1 = [], [], []
    equivalent_vars_by_file = {}
    per_paper_f1 = {}

    if split == "test":
        doi_to_paper_id = _load_test_pubmed_lookup(path2gt)
    else:
        doi_to_paper_id = None

    pubmed_dir = os.path.join(path2gt, "pubmed")

    for eval_conf in conf:
        cs, ps, gs = [], [], []
        precs, recs, f1s = [], [], []

        pred_vars, gt_vars = {}, {}
        equivalent_vars_by_file[eval_conf] = {}
        per_paper_f1[eval_conf] = {} 

        for fname in sorted(os.listdir(pubmed_dir)):
            if not fname.endswith(".json"):
                continue

            if split == "test":
                # Mirrors your original test-set filtering logic.
                # Keep only files that contain exactly one underscore.
                if fname.count("_") != 1:
                    continue

            gt_path = os.path.join(pubmed_dir, fname)

            with open(gt_path, "r") as f:
                extractions = json.load(f)

            gt_rels = extractions["relations"]
            doi_url = extractions.get("source")

            if split == "test":
                paper_name_id = doi_to_paper_id.get(doi_url)

                if paper_name_id is None:
                    equivalent_vars_by_file[eval_conf][fname] = {}
                    continue
            else:
                paper_name_id = fname.replace(".json", "")

            pred_path = os.path.join(path2preds, f"{paper_name_id}{suffix}")

            if os.path.exists(pred_path):
                with open(pred_path, "r") as f:
                    preds = json.load(f)

                if relations_field:
                    preds = preds["relations"]

                pred_rels = preds
            else:
                print(f"[ERROR] Did not find predictions for:", paper_name_id)

                precs.append(0)
                recs.append(0)
                f1s.append(0)
                cs.append(0)
                ps.append(0)
                gs.append(len(gt_rels))
                equivalent_vars_by_file[eval_conf][fname] = {}
                per_paper_f1[eval_conf][fname] = 0 
                continue

            gt_vars[fname] = set(
                [t["Construct from"].lower().strip() for t in gt_rels if t["Construct from"]]
                + [t["Construct to"].lower().strip() for t in gt_rels if t["Construct to"]]
                + [
                    t["Construct Moderator"].lower().strip()
                    for t in gt_rels
                    if t["Construct Moderator"]
                ]
            )

            pred_vars[fname] = set(
                [t["cause"].lower().strip() for t in pred_rels]
                + [t["effect"].lower().strip() for t in pred_rels]
                + [
                    t["moderator"].lower().strip()
                    for t in pred_rels
                    if "moderator" in t and t["moderator"]
                ]
                + [
                    t["mediator"].lower().strip()
                    for t in pred_rels
                    if "mediator" in t and t["mediator"]
                ]
            )

            if mapping_dict and fname in mapping_dict:
                paper_mapping = mapping_dict[fname]
            else:
                paper_mapping = None

            if eval_conf == "3-tuple":
                nc, np_, ng, p, r, f1, equivalent_vars = eval(
                    pred_rels,
                    gt_rels,
                    human_eval=human_eval,
                    mapping_dict=paper_mapping,
                    include_pc=False,
                    include_sig=False,
                    ignore_imgs=False,
                    var_threshold=var_match,
                )
            elif eval_conf == "4-tuple":
                nc, np_, ng, p, r, f1, equivalent_vars = eval(
                    pred_rels,
                    gt_rels,
                    human_eval=human_eval,
                    mapping_dict=paper_mapping,
                    include_pc=True,
                    include_sig=False,
                    ignore_imgs=False,
                    var_threshold=var_match,
                )
            elif eval_conf == "5-tuple":
                nc, np_, ng, p, r, f1, equivalent_vars = eval(
                    pred_rels,
                    gt_rels,
                    human_eval=human_eval,
                    mapping_dict=paper_mapping,
                    include_pc=True,
                    include_sig=True,
                    ignore_imgs=False,
                    var_threshold=var_match,
                )
            else:
                raise ValueError(f"Unknown eval config: {eval_conf}")

            equivalent_vars_by_file[eval_conf][fname] = equivalent_vars

            if nc is None:
                continue

            precs.append(p)
            recs.append(r)
            f1s.append(f1)
            cs.append(nc)
            ps.append(np_)
            gs.append(ng)
            per_paper_f1[eval_conf][fname] = f1

        prec, rec, micro_f1 = _safe_micro_prf(sum(cs), sum(ps), sum(gs))

        out_prec.append(round(prec, 4))
        out_rec.append(round(rec, 4))
        out_f1.append(round(micro_f1, 4))

        if verbose:
            print(100 * "=")
            print(f"RESULTS for {eval_conf}:")
            print("Precision:", round(prec, 4))
            print("Recall:", round(rec, 4))
            print("F1:", round(micro_f1, 4))

            if len(precs) > 0:
                print("Macro Precision:", round(sum(precs) / len(precs), 4))
                print("Macro Recall:", round(sum(recs) / len(recs), 4))
                print("Macro F1:", round(sum(f1s) / len(f1s), 4))

            print(150 * "=")

    return out_rec, out_prec, out_f1, equivalent_vars_by_file, per_paper_f1


def _mean_std_str(values, round_mean=2, round_std=5):
    """
    Format values as mean ± std.
    """
    mean = round(float(np.mean(values)), round_mean)
    std = round(float(np.std(values)), round_std)
    return f"{mean} ± {std}"


def summarize_eval_results(
    model_dirs,
    config=CONFIG,
    n_runs=5,
    split="val",
    base_path=None,
    path2gt=None,
    mapping_dict=None,
    relations_field=True,
    var_match=0.9,
    round_mean=2,
    round_std=5,
):
    """
    Evaluate multiple model directories on both datasets, for both regular and ensemble outputs.

    Parameters:
      model_dirs:
        List of prediction directory names.

      split:
        Either "val" or "test".

      base_path:
        Directory containing model prediction folders.
        If None, defaults to f"../{split}".

      path2gt:
        Directory containing ground-truth labels.
        If None, defaults to f"../{split}".

    Returns:
      pandas.DataFrame with grouped columns:
        - Bo8: Precision, Recall, F1
        - PubMed: Precision, Recall, F1
    """
    _validate_split(split)

    if base_path is None:
        base_path = f"../{split}"

    if path2gt is None:
        path2gt = f"../{split}"

    rows = []
    per_paper_scores = {}

    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)

        if model_dir == "output_r1_ensemble":
            eval_modes = [
                {
                    "model_name": model_dir,
                    "suffixes": ["_ensemble_cleaned.json" for _ in range(1, 4)],
                    "relations_field": False,
                },
            ]

        elif "r1-distill" in model_dir:
            eval_modes = [
                {
                    "model_name": model_dir,
                    "suffixes": [f"_preds_temp=0.1_run{i}.json" for i in range(1, n_runs + 1)],
                    "relations_field": True,
                },
            ]
        elif "qwen3.5" in model_dir:
            eval_modes = [
                {
                    "model_name": model_dir,
                    "suffixes": [f"_temp=0.1_run{i}_cleaned.json" for i in range(1, n_runs + 1)],
                    "relations_field": True,
                },
            ]
            
        elif "r1" in model_dir:
            eval_modes = [
                {
                    "model_name": model_dir,
                    "suffixes": [f"_run{i}_cleaned.json" for i in range(1, 2)],
                    "relations_field": False,
                },
            ]

        elif "_oai_" in model_dir:
            eval_modes = [
                {
                    "model_name": model_dir,
                    "suffixes": [f".json" for i in range(1, 2)],
                    "relations_field": False,
                },
            ]

        elif "_ArrowRCNN_" in model_dir:
            eval_modes = [
                {
                    "model_name": model_dir,
                    "suffixes": [f"_cleaned.json" for i in range(1, 2)],
                    "relations_field": False,
                },
            ]

        else:
            eval_modes = [
                {
                    "model_name": model_dir,
                    "suffixes": [
                        f"_run{i}_cleaned.json"
                        for i in range(1, n_runs + 1)
                    ],
                    "relations_field": relations_field,
                },
            ]

        for mode in eval_modes:
            model_name = mode["model_name"]
            suffixes = mode["suffixes"]

            disknet_recall, disknet_precision, disknet_f1 = [], [], []
            pubmed_recall, pubmed_precision, pubmed_f1 = [], [], []

            for run_idx, suffix in enumerate(suffixes):
                print(suffix, model_path)

                if mapping_dict and f"run{run_idx+1}" in mapping_dict.keys():
                    run_mapping_dict = mapping_dict[f"run{run_idx+1}"]
                else:
                    run_mapping_dict = mapping_dict

                disknet_res = eval_disknet(
                    model_path,
                    suffix,
                    config,
                    split=split,
                    mapping_dict=run_mapping_dict,
                    verbose=False,
                    relations_field=mode["relations_field"],
                    var_match=var_match,
                    path2gt=path2gt,
                )

                plot_f1_histogram(disknet_res[-1], title="Bo8 paper-level F1")
                print_low_f1(disknet_res[-1])
                print_perfect_f1(disknet_res[-1])

                disknet_rec, disknet_prec, disknet_f1_scores = disknet_res[:3]

                disknet_recall.append(disknet_rec)
                disknet_precision.append(disknet_prec)
                disknet_f1.append(disknet_f1_scores)

                pubmed_res = eval_pubmed(
                    model_path,
                    suffix,
                    config,
                    split=split,
                    mapping_dict=run_mapping_dict,
                    verbose=False,
                    relations_field=mode["relations_field"],
                    var_match=var_match,
                    path2gt=path2gt,
                )
                
                plot_f1_histogram(pubmed_res[-1], title="PubMed paper-level F1")
                print_low_f1(pubmed_res[-1])
                print_perfect_f1(pubmed_res[-1])

                pubmed_rec, pubmed_prec, pubmed_f1_scores = pubmed_res[:3]

                pubmed_recall.append(pubmed_rec)
                pubmed_precision.append(pubmed_prec)
                pubmed_f1.append(pubmed_f1_scores)

            for idx, eval_config in enumerate(config):
                rows.append(
                    {
                        ("Model", ""): model_name,
                        ("Eval config", ""): eval_config,

                        ("Bo8", "Precision"): _mean_std_str(
                            [x[idx] for x in disknet_precision],
                            round_mean=round_mean,
                            round_std=round_std,
                        ),
                        ("Bo8", "Recall"): _mean_std_str(
                            [x[idx] for x in disknet_recall],
                            round_mean=round_mean,
                            round_std=round_std,
                        ),
                        ("Bo8", "F1"): _mean_std_str(
                            [x[idx] for x in disknet_f1],
                            round_mean=round_mean,
                            round_std=round_std,
                        ),

                        ("PubMed", "Precision"): _mean_std_str(
                            [x[idx] for x in pubmed_precision],
                            round_mean=round_mean,
                            round_std=round_std,
                        ),
                        ("PubMed", "Recall"): _mean_std_str(
                            [x[idx] for x in pubmed_recall],
                            round_mean=round_mean,
                            round_std=round_std,
                        ),
                        ("PubMed", "F1"): _mean_std_str(
                            [x[idx] for x in pubmed_f1],
                            round_mean=round_mean,
                            round_std=round_std,
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    df = df.set_index([("Model", ""), ("Eval config", "")])
    df.index.names = ["Model", "Eval config"]

    return df
