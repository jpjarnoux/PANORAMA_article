#!/usr/bin/env python3
# coding: utf-8

# Default libraries
from typing import Dict, FrozenSet, Set, Tuple
from pathlib import Path

# Installed libraries
from itertools import chain
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib_venn import venn3

# Local libraries
from sources.dev_utils import similar


def compare_sets(A: Set[FrozenSet[str]], B: Set[FrozenSet[str]], C: Set[FrozenSet[str]], threshold: float) -> Dict:
    """Compare three sets and find unique and common elements among them.

    Args:
        A (Set[FrozenSet[str]]): First set of frozensets to compare.
        B (Set[FrozenSet[str]]): Second set of frozensets to compare.
        C (Set[FrozenSet[str]]): Third set of frozensets to compare.
        threshold (float): Threshold for similarity.

    Returns:
        Dict: A dictionary with counts and frozensets categorized by their intersections.
    """
    # Create unique sets from nested lists
    unique_A = set(chain(*A))
    unique_B = set(chain(*B))
    unique_C = set(chain(*C))

    # Find intersections
    common_ABC = unique_A & unique_B & unique_C
    common_AB = (unique_A & unique_B) - common_ABC
    common_AC = (unique_A & unique_C) - common_ABC
    common_BC = (unique_B & unique_C) - common_ABC

    # Find unique elements
    unique_A -= (common_ABC | common_AB | common_AC)
    unique_B -= (common_ABC | common_AB | common_BC)
    unique_C -= (common_ABC | common_AC | common_BC)

    def find_frozensets(category: Set[str]) -> Dict[str, Set[FrozenSet[str]]]:
        """Find frozensets associated with each category.

        Args:
            category (Set[str]): The set of elements to find.

        Returns:
            Dict[str, Set[FrozenSet[str]]]: A dictionary of frozensets categorized by their presence in the sets.
        """
        mapping = {"A": A, "B": B, "C": C}
        result = {k: set() for k in mapping}

        for key, group in mapping.items():
            for fs in group:
                if category & fs:
                    result[key].add(fs)

        return result

    # Build the dictionary of frozensets
    frozenset_dict = {
        "111": find_frozensets(common_ABC),
        "110": find_frozensets(common_AB),
        "101": find_frozensets(common_AC),
        "011": find_frozensets(common_BC),
        "100": find_frozensets(unique_A),
        "010": find_frozensets(unique_B),
        "001": find_frozensets(unique_C),
    }

    # Return the results
    return {
        "counts": {
            "111": len(common_ABC),
            "110": len(common_AB),
            "101": len(common_AC),
            "011": len(common_BC),
            "100": len(unique_A),
            "010": len(unique_B),
            "001": len(unique_C),
        },
        "frozensets": frozenset_dict,
    }


def get_common_specified_systems(padloc: Dict[str, Set[FrozenSet[str]]], dfinder: Dict[str, Set[FrozenSet[str]]],
                                 panorama: Dict[str, Set[FrozenSet[str]]],
                                 threshold: float = 0.8) -> Dict[str, int]:
    """Calculate common and specific systems among three tools.

    Args:
        padloc (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PADLOC.
        dfinder (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for DFinder.
        panorama (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PANORAMA.
        threshold (float): Threshold for similarity.

    Returns:
        Dict: A dictionary containing subset counts.
    """
    sys_type_set = set(padloc.keys()).union(dfinder.keys(), panorama.keys())

    subset_dict = {x: 0 for x in ["100", "010", "110", "001", "101", "011", "111"]}

    for sys_type in sys_type_set:
        padloc_systems = padloc.get(sys_type, set())
        dfinder_systems = dfinder.get(sys_type, set())
        panorama_systems = panorama.get(sys_type, set())

        comp_dict = compare_sets(padloc_systems, dfinder_systems, panorama_systems, threshold)

        for k, v in comp_dict['counts'].items():
            subset_dict[k] += v

    return subset_dict


def plot_venn_diagram(padloc: Dict[str, Set[FrozenSet[str]]], dfinder: Dict[str, Set[FrozenSet[str]]],
                      panorama: Dict[str, Set[FrozenSet[str]]], output: Path, threshold: float = 0.5) -> None:
    """Generate and display a Venn diagram comparing the systems detected by different tools.

    Args:
        padloc (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PADLOC.
        dfinder (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for DFinder.
        panorama (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PANORAMA.
        output (Path): Path to save the Venn diagram.
        threshold (float): Threshold for similarity.
    """
    n_padloc = sum([len(x) for x in padloc.values()])
    n_df = sum([len(x) for x in dfinder.values()])
    n_panorama = sum([len(x) for x in panorama.values()])
    print(f"\nNumber of pangenome systems: PADLOC {n_padloc}, DFINDER {n_df}, PANORAMA {n_panorama}")
    subset_dict = get_common_specified_systems(padloc, dfinder, panorama, threshold)

    # Create a figure and axis for the Venn diagram
    fig, ax = plt.subplots(figsize=(20, 11.25))

    # Generate the Venn diagram
    venn = venn3(subsets=subset_dict, set_labels=(f'PADLOC ({n_padloc})', f'DefenseFinder ({n_df})',
                                                  f'PANORAMA ({n_panorama})'))
    for label in venn.set_labels:
        if label:
            label.set_fontsize(20)

    for subset in venn.subset_labels:
        if subset:
            subset.set_fontsize(14)

    print(f'PADLOC: {subset_dict["100"]}, DFinder: {subset_dict["010"]}, PANORAMA: {subset_dict["001"]}, '
          f'PADLOC & DFinder: {subset_dict["110"]}, PANORAMA & PADLOC: {subset_dict["101"]}, '
          f'PANORAMA & DFinder: {subset_dict["011"]}, ALL: {subset_dict["111"]}')

    plt.savefig(output / "Venn.png", transparent=False)
    plt.savefig(output / "Venn.svg", transparent=False, format="svg")
    plt.show()


def _build_similarity_graph(set1: Set[FrozenSet[str]], set2: Set[FrozenSet[str]], threshold: float) -> nx.Graph:
    """Build a similarity graph where elements of set1 and set2 are connected if they are similar.

    Args:
        set1 (Set[FrozenSet[str]]): First set of frozensets to compare.
        set2 (Set[FrozenSet[str]]): Second set of frozensets to compare.
        threshold (float): Threshold for similarity.

    Returns:
        nx.Graph: A networkx graph with nodes connected based on similarity.
    """
    G = nx.Graph()

    # Add nodes
    for item in set1:
        G.add_node((item, "set1"))  # Mark each node by its origin set
    for item in set2:
        G.add_node((item, "set2"))

    # Add edges for similar elements
    for item1 in set1:
        for item2 in set2:
            if similar(item1, item2, threshold):
                G.add_edge((item1, "set1"), (item2, "set2"))

    return G


def _compute_common_and_specific(set1: Set[FrozenSet[str]], set2: Set[FrozenSet[str]],
                                 threshold: float) -> Tuple[int, int, int]:
    """Analyze connected components of the graph to determine common and specific elements.

    Args:
        set1 (Set[FrozenSet[str]]): First set of frozensets to compare.
        set2 (Set[FrozenSet[str]]): Second set of frozensets to compare.
        threshold (float): Threshold for similarity.
        err (str): Error message prefix for debugging.

    Returns:
        tuple: Counts of common groups and specific elements in set1 and set2.
    """
    G = _build_similarity_graph(set1, set2, threshold)
    components = list(nx.connected_components(G))

    common_count = 0
    specific_count_set1 = 0
    specific_count_set2 = 0

    for component in components:
        has_set1 = any(node[1] == "set1" for node in component)
        has_set2 = any(node[1] == "set2" for node in component)

        if has_set1 and has_set2:
            common_count += 1  # Mixed group = common
        elif has_set1:
            specific_count_set1 += 1  # Group only in set1
        elif has_set2:
            specific_count_set2 += 1  # Group only in set2

    return common_count, specific_count_set1, specific_count_set2


def count_system_similar(padloc: Dict[str, Set[FrozenSet[str]]],
                         dfinder: Dict[str, Set[FrozenSet[str]]],
                         panorama_padloc: Dict[str, Set[FrozenSet[str]]],
                         panorama_dfinder: Dict[str, Set[FrozenSet[str]]],
                         output: Path, threshold: float = 0.5) -> None:
    """Count similar systems among different tools and save the results to a CSV file.

    Args:
        padloc (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PADLOC.
        dfinder (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for DFinder.
        panorama_padloc (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PANORAMA compared with PADLOC.
        panorama_dfinder (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PANORAMA compared with DFinder.
        output (Path): Path to save the results.
        threshold (float): Threshold for similarity.
    """
    sys_type_set = set(padloc) | set(dfinder) | set(panorama_padloc) | set(panorama_dfinder)
    nb_padloc = len(list(chain(*padloc.values())))
    nb_dfinder = len(list(chain(*dfinder.values())))
    nb_panorama_padloc = len(list(chain(*panorama_padloc.values())))
    nb_panorama_dfinder = len(list(chain(*panorama_dfinder.values())))
    comp_dict = {
        "PADLOC & DFinder": {"Common": 0, "Specific PADLOC": 0, "Specific DFinder": 0},
        "PADLOC & PANORAMA": {"Common": 0, "Specific PADLOC": 0, "Specific PANORAMA": 0},
        "DFinder & PANORAMA": {"Common": 0, "Specific DFinder": 0, "Specific PANORAMA": 0}
    }

    for sys_type in sys_type_set:
        padloc_systems = padloc.get(sys_type, set())
        dfinder_systems = dfinder.get(sys_type, set())
        panorama_padloc_systems = panorama_padloc.get(sys_type, set())
        panorama_dfinder_systems = panorama_dfinder.get(sys_type, set())

        # Comparison PADLOC <-> DFinder
        common, specific_padloc, specific_dfinder = _compute_common_and_specific(padloc_systems, dfinder_systems,
                                                                                 threshold)
        comp_dict["PADLOC & DFinder"]["Common"] += common
        comp_dict["PADLOC & DFinder"]["Specific PADLOC"] += specific_padloc
        comp_dict["PADLOC & DFinder"]["Specific DFinder"] += specific_dfinder

        # Comparison PADLOC <-> PANORAMA
        common, specific_padloc, specific_panorama = _compute_common_and_specific(padloc_systems,
                                                                                  panorama_padloc_systems,
                                                                                  threshold)
        comp_dict["PADLOC & PANORAMA"]["Common"] += common
        comp_dict["PADLOC & PANORAMA"]["Specific PADLOC"] += specific_padloc
        comp_dict["PADLOC & PANORAMA"]["Specific PANORAMA"] += specific_panorama

        # Comparison DFinder <-> PANORAMA
        common, specific_dfinder, specific_panorama = _compute_common_and_specific(dfinder_systems,
                                                                                   panorama_dfinder_systems,
                                                                                   threshold)
        comp_dict["DFinder & PANORAMA"]["Common"] += common
        comp_dict["DFinder & PANORAMA"]["Specific DFinder"] += specific_dfinder
        comp_dict["DFinder & PANORAMA"]["Specific PANORAMA"] += specific_panorama

    res_df = pd.DataFrame.from_dict(comp_dict, orient="index")
    res_df.index = [f"PADLOC ({nb_padloc}) & DFinder ({nb_dfinder})",
                    f"PADLOC ({nb_padloc}) & PANORAMA ({nb_panorama_padloc})",
                    f"DFinder ({nb_dfinder}) & PANORAMA ({nb_panorama_dfinder})"]
    res_df.to_csv(output / "prediction_comparison.csv", index=True)
    print(res_df)
