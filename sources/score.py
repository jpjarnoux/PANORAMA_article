#!/usr/bin/env python3
# coding:utf-8

# default libraries
from pathlib import Path
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple, Union

# installed libraries
import pandas as pd
import matplotlib.pyplot as plt

# local libraries
from sources.dev_utils import type2category, similar


def write_df(data, out_name, grouped: bool = False):
    """
    Writes a DataFrame to a CSV file with optional sorting and grouping.

    Args:
        data: The data to be written, expected to be a list or array-like object.
        out_name: The output filename or path.
        grouped: Boolean indicating whether to group the DataFrame by specific columns.

    Returns:
        None
    """
    df = pd.DataFrame(data)

    if not df.empty:
        # If grouping is required, group by columns 1 and 2, and aggregate column 0
        if grouped:
            df_grouped = df.groupby([1, 2], as_index=False).agg({0: ', '.join})
            df_grouped = df_grouped[[0, 1, 2]]

            # Sort the grouped DataFrame by columns 2 and 1
            df_sorted = df_grouped.sort_values(by=[2, 1], ascending=[True, False])
        else:
            # Sort the DataFrame by columns 1 and 0
            df_sorted = df.sort_values(by=[1, 0], ascending=[True, False])

        # Write the sorted DataFrame to a CSV file
        df_sorted.to_csv(out_name, sep='\t', index=False, header=True)

    else:
        # Write the empty DataFrame directly to a CSV file
        df.to_csv(out_name, sep='\t', index=False, header=True)


def compute_score(systems2pred: Dict[str, List[int]]) -> Tuple[float, float, float]:
    """
    Calculate recall, precision, and F1-score for each system type

    Args:
        systems2pred: Dictionary of system type to prediction comparison results

    Returns:
        Dict[str, List[float]]: For each system type the recall, precision and F1-score
    """
    total = [0, 0, 0]  # To track total true positives, false negatives, and false positives
    scores = defaultdict(list)
    for system_type, pred in systems2pred.items():
        true_positive, false_negative, false_positive = pred
        total = [x + y for x, y in zip(total, pred)]
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        f1_score = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0
        scores["systems"].append(system_type)
        scores["recall"].append(recall * 100)
        scores["precision"].append(precision * 100)
        scores["f1_score"].append(f1_score * 100)
        scores["count"].append(true_positive + false_negative + false_positive)

    # Calculate total recall, precision, and F1-score across all system types
    tot_recall = total[0] / (total[0] + total[1]) if total[0] + total[1] > 0 else 0
    tot_precision = total[0] / (total[0] + total[2]) if total[0] + total[2] > 0 else 0
    tot_f1_score = 2 * (tot_recall * tot_precision) / (
            tot_recall + tot_precision) if tot_recall + tot_precision > 0 else 0

    return tot_recall, tot_precision, tot_f1_score


def compute_score_pangenome(panorama: Dict[str, Set[FrozenSet[str]]], other: Dict[str, Set[FrozenSet[str]]],
                            other_name: str, output: Path, threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F1-score for the comparison of two pangenomes.

    Args:
        panorama: Dictionary of systems from the panorama with set of frozen sets.
        other: Dictionary of systems to be compared against the panorama.
        other_name: Name of the other system for output file labeling.
        output: Path where output files will be saved.

    Returns:
        scores: Dictionary containing recall, precision, F1-score, and counts for each system type.
    """
    other_no_match_set = set()
    other_no_match = []
    panorama_no_match_set = set()
    panorama_no_match = []
    systems2pred = defaultdict(lambda: [0, 0, 0])  # TP, FN, FP
    valid_sys = set()
    for system_type in other.keys():
        if system_type in panorama:
            panorama_systems = panorama[system_type]
            other_systems = list(other[system_type])
            other_index = 0
            seen = set()
            # Iterate through all other systems to find matches with panorama systems
            while other_index < len(other_systems):
                other_sys = other_systems[other_index]
                common = False
                for panorama_sys in panorama_systems:
                    inter = other_sys.intersection(panorama_sys)
                    if similar(panorama_sys, other_sys, threshold):
                        # systems2pred[system_type][0] += 1  # True positive
                        systems2pred[system_type][0] += len(inter)  # True positive
                        common = True
                        seen.add(frozenset(panorama_sys))
                        valid_sys.add(frozenset(panorama_sys))
                if not common:
                    systems2pred[system_type][1] += len(other_sys)  # False negative
                    # systems2pred[system_type][1] += 1  # False negative
                    line = [system_type, ", ".join(sorted(list(map(str, other_sys))))]
                    frozen_line = frozenset(line)
                    if frozen_line not in other_no_match_set:
                        other_no_match_set.add(frozen_line)
                        other_no_match.append(line)
                other_index += 1

            # Find unmatched panorama systems
            if len(panorama_systems) - len(seen) > 0:
                for panorama_sys in panorama_systems:
                    if panorama_sys not in seen:  # and panorama_sys not in valid_sys:
                        systems2pred[system_type][2] += len(panorama_sys)  # False positive
                        # systems2pred[system_type][2] += 1  # False positive
                        line = [system_type, ", ".join(sorted(list(map(str, panorama_sys))))]
                        frozen_line = frozenset(line)
                        if frozen_line not in panorama_no_match_set:
                            panorama_no_match_set.add(frozen_line)
                            panorama_no_match.append(line)
        else:
            systems2pred[system_type][1] += sum(len(sys) for sys in other[system_type])  # False negative
            # systems2pred[system_type][1] += len(other[system_type])  # False negative
            for system in other[system_type]:
                line = [system_type, ", ".join(sorted(list(map(str, system))))]
                frozen_line = frozenset(line)
                if frozen_line not in other_no_match_set:
                    other_no_match_set.add(frozen_line)
                    other_no_match.append(line)

    # Find panorama-only systems that are not in 'other'
    for system_type in set(panorama.keys()).difference(set(other.keys())):
        for panorama_sys in panorama[system_type]:
            # if panorama_sys not in valid_sys:
            systems2pred[system_type][2] += 1  # False positive
            line = [system_type, ", ".join(sorted(list(map(str, panorama_sys))))]
            frozen_line = frozenset(line)
            if frozen_line not in panorama_no_match_set:
                panorama_no_match_set.add(frozen_line)
                panorama_no_match.append(line)

    # Write unmatched systems to output files
    write_df(other_no_match, output / f'{other_name}_miss_pangenome.tsv')
    write_df(panorama_no_match, output / f'PANORAMA_{other_name}_miss_pangenome.tsv')

    return compute_score(systems2pred)


def compute_score_organisms(panorama_res: Dict[str, Dict[str, List[Set[str]]]],
                            other: Dict[str, Dict[str, List[Set[str]]]],
                            other_name: str, output: Path, threshold: float = 0.5
                            ) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F1-score for organisms by comparing systems between panorama and other results.

    Args:
        panorama_res: Dictionary of systems for each organism in the panorama result.
        other: Dictionary of systems for each organism in the other result.
        other_name: Name of the other result for output file labeling.
        output: Path where output files will be saved.

    Returns:
        scores: Dictionary containing recall, precision, F1-score, and counts for each system type across organisms.
    """
    other_no_match_set = set()
    other_no_match = []
    panorama_no_match_set = set()
    panorama_no_match = []
    systems2pred = defaultdict(lambda: [0, 0, 0])  # TP, FN, FP
    valid_sys = set()

    for organism, systems_dict in other.items():
        panorama_sys_name = set(panorama_res.get(organism, {}).keys())
        for system_type, systems in systems_dict.items():
            if system_type in panorama_res[organism]:
                panorama_sys_name.remove(system_type)
                panorama_systems = panorama_res[organism][system_type]
                other_index = 0
                seen = set()
                # Iterate through all other systems to find matches with panorama systems
                while other_index < len(systems):
                    other_sys = systems[other_index]
                    common = False
                    for panorama_sys in panorama_systems:
                        inter = other_sys.intersection(panorama_sys)
                        if similar(other_sys, panorama_sys, threshold):
                            systems2pred[system_type][0] += 1  # True positive
                            common = True
                            seen.add(frozenset(panorama_sys))
                            valid_sys.add(frozenset(panorama_sys))

                    if not common:
                        systems2pred[system_type][1] += len(other_sys)  # False negative
                        # systems2pred[system_type][1] += 1  # False negative
                        line = [organism, system_type, ", ".join(sorted(list(map(str, other_sys))))]
                        frozen_line = frozenset(line)
                        if frozen_line not in other_no_match_set:
                            other_no_match_set.add(frozen_line)
                            other_no_match.append(line)
                    other_index += 1

                # Find unmatched panorama systems
                if len(panorama_systems) - len(seen) > 0:
                    for panorama_sys in panorama_systems:
                        if panorama_sys not in seen:  # and panorama_sys not in valid_sys:
                            systems2pred[system_type][2] += len(panorama_sys)  # False positive
                            # systems2pred[system_type][2] += 1  # False positive
                            line = [organism, system_type, ", ".join(sorted(list(map(str, panorama_sys))))]
                            frozen_line = frozenset(line)
                            if frozen_line not in panorama_no_match_set:
                                panorama_no_match_set.add(frozen_line)
                                panorama_no_match.append(line)
            else:
                systems2pred[system_type][1] += sum(len(sys) for sys in systems)  # False negative
                # systems2pred[system_type][1] += len(other[system_type])  # False negative
                for system in systems:
                    line = [organism, system_type, ", ".join(sorted(list(map(str, system))))]
                    frozen_line = frozenset(line)
                    if frozen_line not in other_no_match_set:
                        other_no_match_set.add(frozen_line)
                        other_no_match.append(line)

        # Find panorama-only systems that are not in 'other'
        for system_type in panorama_sys_name:
            for system in panorama_res[organism][system_type]:
                # if system not in valid_sys:
                systems2pred[system_type][2] += 1  # False positive
                line = [organism, system_type, ", ".join(sorted(list(map(str, system))))]
                frozen_line = frozenset(line)
                if frozen_line not in panorama_no_match_set:
                    panorama_no_match_set.add(frozen_line)
                    panorama_no_match.append(line)

    # Write unmatched systems to output files
    write_df(other_no_match, output / f'{other_name}_miss_organism.tsv', True)
    write_df(panorama_no_match, output / f'PANORAMA_{other_name}_miss_organism.tsv', True)

    return compute_score(systems2pred)


def parse_scores(score_padloc: Dict[str, List[Union[str, float]]],
                 score_dfinder: Dict[str, List[Union[str, float]]]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def get_score(dict_score: Dict[str, List[Union[str, float]]], sys_name: str, metric: str) -> float:
        try:
            score = dict_score[metric][dict_score["systems"].index(sys_name)]
        except ValueError:
            return 0
        else:
            return score

    recall_score = {"systems": [], "PADLOC": [], "Defense-Finder": []}
    precision_score = {"systems": [], "PADLOC": [], "Defense-Finder": []}
    f1_score = {"systems": [], "PADLOC": [], "Defense-Finder": []}
    count = {"systems": [], "PADLOC": [], "Defense-Finder": []}

    recall_df, precision_df, f1_df, count_df = None, None, None, None
    for system in sorted(set(score_padloc['systems']).union(set(score_dfinder["systems"]))):
        recall_score["systems"].append(system)
        recall_score["PADLOC"].append(get_score(score_padloc, system, "recall"))
        recall_score["Defense-Finder"].append(get_score(score_dfinder, system, "recall"))
        recall_df = pd.DataFrame.from_dict(recall_score).set_index('systems')

        precision_score["systems"].append(system)
        precision_score["PADLOC"].append(get_score(score_padloc, system, "precision"))
        precision_score["Defense-Finder"].append(get_score(score_dfinder, system, "precision"))
        precision_df = pd.DataFrame.from_dict(precision_score).set_index('systems')

        f1_score["systems"].append(system)
        f1_score["PADLOC"].append(get_score(score_padloc, system, "f1_score"))
        f1_score["Defense-Finder"].append(get_score(score_dfinder, system, "f1_score"))
        f1_df = pd.DataFrame.from_dict(f1_score).set_index('systems')

        count["systems"].append(system)
        count["PADLOC"].append(get_score(score_padloc, system, "count"))
        count["Defense-Finder"].append(get_score(score_dfinder, system, "count"))
        count_df = pd.DataFrame.from_dict(count).set_index('systems')
    return recall_df, precision_df, f1_df, count_df


def group_systems(scores: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
                  groups: Tuple[Set[str], Set[str]] = None):
    df_count = scores[3]
    df_count_grouped = df_count.groupby(type2category).sum()
    df_count_grouped = df_count_grouped.groupby(lambda idx: 'other' if idx in groups[1] else idx).sum()
    sd_scores = []
    for i in range(0, 3):
        score = scores[i]
        df_score = pd.DataFrame.from_dict(score)
        df_score.index = df_count.index
        df_sd = (df_score * df_count).groupby(type2category).sum().groupby(
            lambda idx: 'other' if idx in groups[1] else idx).sum().divide(df_count_grouped)

        sd_scores.append(df_sd)
    return sd_scores


def gen_barplot(scores: pd.DataFrame, measure: str, figname: str, output: Path) -> None:
    ax = scores.plot.bar(layout="constrained", figsize=(20, 11.25))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f'{measure} (%)')
    ax.set_title(f'Panorama {measure} in comparison to PADLOC and Defense finder')
    ax.set_xticks(ax.get_xticks(), labels=scores.index, rotation=45, ha='right', rotation_mode='anchor')
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 105)
    for c in ax.containers:
        # Filter the labels
        labels = [round(v, 2) if 0 < v < 100 else "" for v in c.datavalues]
        ax.bar_label(c, labels=labels, padding=5)
    plt.savefig(output / figname, transparent=False)
    return ax


def plot_scores(scores_padloc, scores_dfinder, main_group, other_group, output, suffix: str = ""):
    axes = []
    scores = parse_scores(scores_padloc, scores_dfinder)
    # gen_boxplot(scores[-1])
    for idx, measure in enumerate(["recall", "precision", "f1-score"]):
        axes.append(gen_barplot(scores[idx], measure, f'barplot_{measure}{suffix}.png', output))

    group_scores = group_systems(scores, groups=(main_group, other_group))
    for idx, measure in enumerate(["recall", "precision", "f1-score"]):
        axes.append(gen_barplot(group_scores[idx], measure, f'barplot_group_{measure}{suffix}.png', output))
