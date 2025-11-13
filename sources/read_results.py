#!/usr/bin/env python3
# coding:utf-8

# default libraries
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# installed libraries
import pandas as pd
from tqdm import tqdm

# local libraries
from sources.dev_utils import padloc2dfinder, vote_partitions


def load_gene_families(gene_families: str, return_dict: Dict[str, Any], disable_bar: bool = False) -> None:
    """Loads gene families from a file.

    Args:
        gene_families (str): Path to the gene families file.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
        disable_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
    """
    gene2family = {}
    fam2size = defaultdict(lambda: 0)
    with open(gene_families, 'r') as file:
        for row in tqdm(csv.reader(file, delimiter='\t'), disable=disable_bar, desc="Read gene families file"):
            gene2family[str(row[1])] = str(row[0])
            fam2size[str(row[0])] += 1

    # Storing the results in the shared dictionary
    return_dict['gene2family'] = gene2family
    return_dict['fam2size'] = dict(fam2size)


def load_locus_tags(locus2protid_file: Optional[str], return_dict: Dict[str, Any]) -> None:
    """Loads locus tags and protein IDs from a file.

    Args:
        locus2protid_file (Optional[str]): Path to the locus2protid file, or None if not provided.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
    """
    if locus2protid_file:
        locus2protid_df = pd.read_csv(locus2protid_file, sep='\t', header=None, names=['PA', 'NP'])
        locus2protid = locus2protid_df.set_index('NP')['PA'].to_dict()
        return_dict['locus2protID'] = locus2protid
    else:
        return_dict['locus2protID'] = None


def load_padloc_results(padloc_file: Path, gene2family: Dict[str, str], locus2protid: Optional[Dict[str, str]],
                        return_dict: Dict[str, Any], disable_bar: bool = False, position: int = 0) -> None:
    """Loads Padloc results.

    Args:
        padloc_file (str): Path to the Padloc results file.
        gene2family (Dict[str, str]): Gene-to-family mapping.
        locus2protid (Optional[Dict[str, str]]): Locus-to-protein ID mapping.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
        disable_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
        position (int, optional): Position of the progress bar. Defaults to 0.
    """
    padloc_res, count, fam_annot = get_padloc_results(padloc_file, gene2family, locus2protid, disable_bar, position)

    # Storing the Padloc results in the shared dictionary
    return_dict['padloc_res'] = padloc_res
    return_dict['padloc_count'] = count
    return_dict['padloc_fam_annot'] = fam_annot


def load_panorama_results(panorama_file: Path, return_dict: Dict[str, Any], key_prefix: str,
                          disable_bar: bool = False, position: int = 0) -> None:
    """Loads Panorama results.

    Args:
        panorama_file (str): Path to the Panorama results file.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
        key_prefix (str): Prefix for keys in the shared dictionary to differentiate between results.
        disable_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
        position (int, optional): Position of the progress bar. Defaults to 0.
    """
    panorama_res, count, systems2partitions, name2systems, fam_annot = get_panorama_results(panorama_file, disable_bar,
                                                                                            position, key_prefix)

    # Storing Panorama results in the shared dictionary with appropriate key prefixes
    return_dict[f'panorama_{key_prefix}_res'] = panorama_res
    return_dict[f'panorama_{key_prefix}_count'] = count
    return_dict[f'{key_prefix}_systems2partitions'] = systems2partitions
    return_dict[f'{key_prefix}_name2systems'] = name2systems
    return_dict[f'panorama_{key_prefix}_fam_annot'] = fam_annot


def load_dfinder_results(dfinder_file: Path, gene2family: Dict[str, str], locus2protid: Optional[Dict[str, str]],
                         return_dict: Dict[str, Any], disable_bar: bool = False, position: int = 0) -> None:
    """Loads Defense Finder results.

    Args:
        dfinder_file (str): Path to the Defense Finder results file.
        gene2family (Dict[str, str]): Gene-to-family mapping.
        locus2protid (Optional[Dict[str, str]]): Locus-to-protein ID mapping.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
        disable_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
        position (int, optional): Position of the progress bar. Defaults to 0.
    """
    dfinder_res, count, fam_annot = get_df_results(dfinder_file, gene2family, locus2protid, disable_bar, position)

    # Storing Defense Finder results in the shared dictionary
    return_dict['dfinder_res'] = dfinder_res
    return_dict['dfinder_count'] = count
    return_dict['dfinder_fam_annot'] = fam_annot


def get_padloc_results(padloc_dir: Path, gene2families: Dict[str, str], locus2protid: dict = None,
                       disable_bar: bool = False, position: int = 0
                       ) -> Tuple[Dict[str, Dict[str, List[Set[str]]]], Dict[str, int], Dict[str, Set[str]]]:
    """Processes Padloc results to map genes to systems, counts, and annotations.

    Args:
        padloc_dir (Path): Directory containing Padloc result files.
        gene2families (Dict[str, str]): Mapping from genes to families.
        locus2protid (dict, optional): Mapping from locus tags to protein IDs. Defaults to None.
        disable_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
        position (int, optional): Position of the progress bar. Defaults to 0.

    Returns:
        Tuple[Dict[str, Dict[str, List[Set[str]]]], Dict[str, int], Dict[str, Set[str]]]:
            - A mapping of genome to systems with their associated gene families.
            - A count of systems by type.
            - Annotations for each gene family.
    """
    org2system = {}
    systems2count = {}
    fam2annot = {}

    for file in tqdm(list(padloc_dir.rglob('*.csv')), position=position,
                     disable=disable_bar, desc="Read Padloc results"):
        padloc_res = pd.read_csv(file, sep=',', header=0)
        genome = file.stem.replace("_protein_padloc", "")
        org2system[genome] = {}
        for system in padloc_res["system"].unique():
            if not re.search("other", str(system), re.IGNORECASE):
                system_name = padloc2dfinder.get(system, system)
                org2system[genome][system_name] = []
                systems2count[system_name] = 1 if system_name not in systems2count else systems2count[system_name] + 1
                for system_id in padloc_res[padloc_res.system == system]["system.number"].unique():
                    system_family = set()
                    systems2count[system_name] += 1
                    subsys = padloc_res[padloc_res["system.number"] == system_id][["target.name", "protein.name"]]
                    for gene, annot in subsys.itertuples(index=False):
                        prot_id = locus2protid.get(gene, gene) if locus2protid else gene
                        gf = gene2families[str(prot_id)]
                        system_family.add(gf)
                        if gf in fam2annot:
                            fam2annot[gf].add(annot)
                        else:
                            fam2annot[gf] = {annot}
                    org2system[genome][system_name].append(system_family)

    return org2system, systems2count, fam2annot


def get_df_results(dfinder_dir: Path, gene2families: Dict[str, str], locus2protid: dict = None,
                   disable_bar: bool = False, position: int = 0
                   ) -> Tuple[Dict[str, Dict[str, List[Set[str]]]], Dict[str, int], Dict[str, Set[str]]]:
    """Processes Defense Finder (Dfinder) results to map genes to systems, counts, and annotations.

    Args:
        dfinder_dir (Path): Directory containing Dfinder result files.
        gene2families (Dict[str, str]): Mapping from genes to families.
        locus2protid (dict, optional): Mapping from locus tags to protein IDs. Defaults to None.
        disable_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
        position (int, optional): Position of the progress bar. Defaults to 0.

    Returns:
        Tuple[Dict[str, Dict[str, List[Set[str]]]], Dict[str, int], Dict[str, Set[str]]]:
            - A mapping of genome to systems with their associated gene families.
            - A count of systems by type.
            - Annotations for each gene family.
    """

    def extract_prot_id(input_string: str) -> str:
        """Extracts protein ID using regex.

        Args:
            input_string (str): Input string containing protein information.

        Returns:
            str: Extracted locus tag or protein ID.
        """
        match = re.search(r"prot_(\w+\.\d+)", input_string)
        locus_tag = match.group(1)
        return locus2protid.get(locus_tag, locus_tag) if locus2protid else locus_tag

    org2system = {}
    systems2count = {}
    fam2annot = defaultdict(set)

    for file in tqdm(list(dfinder_dir.glob("GCF*/*_translated_cds_defense_finder_genes.tsv")), position=position,
                     disable=disable_bar, desc="Read Defense Finder results"):
        dfinder_res = pd.read_csv(file, sep="\t", header=0)
        genome = file.stem.replace("_translated_cds_defense_finder_genes", "")
        org2system[genome] = {}
        for model in dfinder_res["model_fqn"].unique():
            system = model.split('/')[-1]
            org2system[genome][system] = []
            for system_id in dfinder_res[dfinder_res.model_fqn == model]["sys_id"].unique():
                system_family = set()
                systems2count[system] = 1 if system not in systems2count else systems2count[system] + 1
                subsys = dfinder_res[dfinder_res["sys_id"] == system_id][["hit_id", "gene_name"]]
                for gene, annot in subsys.itertuples(index=False):
                    prot_id = extract_prot_id(gene)
                    gf = gene2families[str(prot_id)]
                    system_family.add(gf)
                    if re.search("__", annot):
                        annot = annot.split("__")[1]
                    if gf in fam2annot:
                        fam2annot[gf].add(annot)
                    else:
                        fam2annot[gf] = {annot}
                org2system[genome][system].append(system_family)

    return org2system, systems2count, fam2annot


def get_panorama_results(panorama_dir: Path, disable_bar: bool = False,
                         position: int = 0, key: str = "",
                         ) -> Tuple[dict, dict, dict, dict, dict]:
    """Processes Panorama results to map genes to systems, partitions, counts, and annotations.

    Args:
        panorama_dir (Path): Directory containing Panorama result files.
        disable_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
        position (int, optional): Position of the progress bar. Defaults to 0.
        key (str, optional): Key to differentiate between Panorama results. Defaults to "".

    Returns:
        Tuple[dict, dict, dict, dict, dict]:
            - Mapping of genome to systems and their gene families.
            - Count of systems by type.
            - Systems to partitions mapping.
            - Name to systems mapping.
            - Annotations for each gene family.
    """

    org2system = {}
    systems2count = {}
    systems2partitions = {}
    name2systems = {}
    fam2annot = {}

    for file in tqdm(list(panorama_dir.glob("*.tsv")), position=position,
                     disable=disable_bar, desc=f"Read Panorama {key} results"):
        panorama_res = pd.read_csv(file, sep='\t', header=0, dtype={"system number": object})
        genome = file.stem
        org2system[genome] = {}

        for system in panorama_res["system name"].unique():
            if not re.search("other", str(system), re.IGNORECASE):
                system_name = padloc2dfinder.get(system, system)
                org2system[genome][system_name] = []
                seen_id = set()

                for system_id in sorted(panorama_res[panorama_res["system name"] == system]["system number"].unique(),
                                        key=lambda x: len(x.split(', ')), reverse=True):
                    if system_id not in seen_id:
                        seen_id.add(system_id)
                        sys_df = panorama_res[panorama_res["system number"] == system_id]
                        if not sys_df[sys_df["annotation"].notna()].empty:
                            system_family = set()
                            partitions = []
                            systems2count[system_name] = 1 if system_name not in systems2count else systems2count[
                                                                                                        system_name] + 1

                            for subsystem_id in sys_df["subsystem number"].unique():
                                subsys_df = sys_df[sys_df["subsystem number"] == subsystem_id]
                                for family, annotation, partition in zip(
                                        subsys_df["gene family"],
                                        subsys_df["annotation"],
                                        subsys_df["partition"],
                                        strict=False
                                ):
                                    if not pd.isna(annotation):
                                        system_family.add(str(family))
                                        partitions.append(partition)
                                        if family in fam2annot:
                                            fam2annot[family].add(annotation)
                                        else:
                                            fam2annot[family] = {annotation}

                            voted_part = vote_partitions(partitions)
                            if system_name in systems2partitions:
                                systems2partitions[system_name].append(voted_part)
                            else:
                                systems2partitions[system_name] = [voted_part]

                            if system_name in name2systems:
                                name2systems[system_name].append(system_family)
                            else:
                                name2systems[system_name] = [system_family]
                            org2system[genome][system_name].append(system_family)

    return org2system, systems2count, systems2partitions, name2systems, fam2annot


def read_panorama_pangenome_systems(systems_file: Path):
    """
    Reads and processes the panorama pangenome systems data from a specified file.

    Args:
        systems_file (Path): Path to the tab-separated file containing the panorama pangenome
            systems data.

    Returns:
        dict: A dictionary where keys are system names and values are sets of frozensets
            of model genomic features.
    """
    systems_df = pd.read_csv(systems_file, sep='\t', header=0, usecols=[0, 1, 4, 6])
    systems_df["model_GF"] = systems_df["model_GF"].str.split(", ").apply(lambda x: frozenset(sorted(x)))
    systems_df["system name"] = systems_df["system name"].apply(lambda x: padloc2dfinder.get(x, x))
    group_df = systems_df.groupby("system name")["model_GF"].apply(set)
    return group_df.to_dict()
