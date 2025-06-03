#!/usr/bin/env python3
# coding:utf-8

# default libraries
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import re
import csv

# installed libraries
from tqdm import tqdm
import pandas as pd

# local libraries
from sources.dev_utils import padloc2dfinder, vote_partitions


def load_gene_families(gene_families: str, return_dict: Dict[str, Any]) -> None:
    """Loads gene families from a file.

    Args:
        gene_families (str): Path to the gene families file.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
    """
    gene2family = {}
    fam2size = defaultdict(lambda: 0)
    with open(gene_families, 'r') as file:
        for row in tqdm(csv.reader(file, delimiter='\t')):
            gene2family[str(row[1])] = str(row[0])
            fam2size[str(row[0])] += 1

    # Storing the results in the shared dictionary
    return_dict['gene2family'] = gene2family
    return_dict['fam2size'] = dict(fam2size)


def load_locus_tags(locus2protID_file: Optional[str], return_dict: Dict[str, Any]) -> None:
    """Loads locus tags and protein IDs from a file.

    Args:
        locus2protID_file (Optional[str]): Path to the locus2protID file, or None if not provided.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
    """
    if locus2protID_file:
        locus2protID_df = pd.read_csv(locus2protID_file, sep='\t', header=None, names=['PA', 'NP'])
        locus2protID = locus2protID_df.set_index('NP')['PA'].to_dict()
        return_dict['locus2protID'] = locus2protID
    else:
        return_dict['locus2protID'] = None


def load_padloc_results(padloc_file: Path, gene2family: Dict[str, str], locus2protID: Optional[Dict[str, str]],
                        return_dict: Dict[str, Any], position: int = 0) -> None:
    """Loads Padloc results.

    Args:
        padloc_file (str): Path to the Padloc results file.
        gene2family (Dict[str, str]): Gene-to-family mapping.
        locus2protID (Optional[Dict[str, str]]): Locus-to-protein ID mapping.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
    """
    padloc_res, count, fam_annot = get_padloc_results(padloc_file, gene2family, locus2protID, position)

    # Storing the Padloc results in the shared dictionary
    return_dict['padloc_res'] = padloc_res
    return_dict['padloc_count'] = count
    return_dict['padloc_fam_annot'] = fam_annot


def load_panorama_results(panorama_file: Path, return_dict: Dict[str, Any], key_prefix: str, position: int = 0) -> None:
    """Loads Panorama results.

    Args:
        panorama_file (str): Path to the Panorama results file.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
        key_prefix (str): Prefix for keys in the shared dictionary to differentiate between results.
    """
    panorama_res, count, systems2partitions, name2systems, fam_annot = get_panorama_results(panorama_file, position,
                                                                                            key_prefix)

    # Storing Panorama results in the shared dictionary with appropriate key prefixes
    return_dict[f'panorama_{key_prefix}_res'] = panorama_res
    return_dict[f'panorama_{key_prefix}_count'] = count
    return_dict[f'{key_prefix}_systems2partitions'] = systems2partitions
    return_dict[f'{key_prefix}_name2systems'] = name2systems
    return_dict[f'panorama_{key_prefix}_fam_annot'] = fam_annot


def load_dfinder_results(dfinder_file: Path, gene2family: Dict[str, str], locus2protID: Optional[Dict[str, str]],
                         return_dict: Dict[str, Any], position: int = 0) -> None:
    """Loads Defense Finder results.

    Args:
        dfinder_file (str): Path to the Defense Finder results file.
        gene2family (Dict[str, str]): Gene-to-family mapping.
        locus2protID (Optional[Dict[str, str]]): Locus-to-protein ID mapping.
        return_dict (Dict[str, Any]): Shared dictionary to store the results.
    """
    dfinder_res, count, fam_annot = get_df_results(dfinder_file, gene2family, locus2protID, position)

    # Storing Defense Finder results in the shared dictionary
    return_dict['dfinder_res'] = dfinder_res
    return_dict['dfinder_count'] = count
    return_dict['dfinder_fam_annot'] = fam_annot


def get_padloc_results(padloc_dir: Path, gene2families: Dict[str, str], locus2protID: dict = None, position: int = 0
                       ) -> Tuple[Dict[str, Dict[str, List[Set[str]]]], Dict[str, int], Dict[str, Set[str]]]:
    """Processes Padloc results to map genes to systems, counts, and annotations.

    Args:
        padloc_dir (Path): Directory containing Padloc result files.
        gene2families (Dict[str, str]): Mapping from genes to families.
        locus2protID (dict, optional): Mapping from locus tags to protein IDs. Defaults to None.

    Returns:
        Tuple[Dict[str, Dict[str, List[Set[str]]]], Dict[str, int], Dict[str, Set[str]]]:
            - A mapping of genome to systems with their associated gene families.
            - A count of systems by type.
            - Annotations for each gene family.
    """
    org2system = {}
    systems2count = {}
    fam2annot = {}

    for file in tqdm(list(padloc_dir.rglob('*.csv')), position=position, desc="Read Padloc results"):
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
                        prot_id = locus2protID.get(gene, gene) if locus2protID else gene
                        gf = gene2families[str(prot_id)]
                        system_family.add(gf)
                        if gf in fam2annot:
                            fam2annot[gf].add(annot)
                        else:
                            fam2annot[gf] = {annot}
                    org2system[genome][system_name].append(system_family)

    return org2system, systems2count, fam2annot


def get_df_results(dfinder_dir: Path, gene2families: Dict[str, str], locus2protID: dict = None, position: int = 0
                   ) -> Tuple[Dict[str, Dict[str, List[Set[str]]]], Dict[str, int], Dict[str, Set[str]]]:
    """Processes Defense Finder (Dfinder) results to map genes to systems, counts, and annotations.

    Args:
        dfinder_dir (Path): Directory containing Dfinder result files.
        gene2families (Dict[str, str]): Mapping from genes to families.
        locus2protID (dict, optional): Mapping from locus tags to protein IDs. Defaults to None.

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
        match = re.search(r'prot_([A-Za-z0-9_]+\.\d+)', input_string)
        locus_tag = match.group(1)
        return locus2protID.get(locus_tag, locus_tag) if locus2protID else locus_tag

    org2system = {}
    systems2count = {}
    fam2annot = defaultdict(set)

    for file in tqdm(list(dfinder_dir.glob("GCF*/*_translated_cds_defense_finder_genes.tsv")), position=position,
                     desc="Read Defense Finder results"):
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


def get_panorama_results(panorama_dir: Path, position: int = 0, key: str = "",
                         ) -> Tuple[dict, dict, dict, dict, dict]:
    """Processes Panorama results to map genes to systems, partitions, counts, and annotations.

    Args:
        panorama_dir (Path): Directory containing Panorama result files.

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

    for file in tqdm(list(panorama_dir.glob("*.tsv")), position=position, desc=f"Read Panorama {key} results"):
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
                    sys_id = system_id.split(', ')
                    if not set(sys_id).issubset(seen_id):
                        regex_sys_id = r'\b' + r'\b|\b'.join(sys_id) + r'\b'
                        seen_id.update(sys_id)
                        sys_df = panorama_res[panorama_res["system number"].str.contains(rf'{regex_sys_id}', na=False)]
                        system_family = set()
                        partitions = []
                        systems2count[system_name] = 1 if system_name not in systems2count else systems2count[
                                                                                                    system_name] + 1

                        for subsystem_id in sys_df["subsystem number"].unique():
                            subsys_df = sys_df[sys_df["subsystem number"] == subsystem_id]
                            for family, annotation, partition in zip(subsys_df["gene family"], subsys_df["annotation"],
                                                                     subsys_df["partition"]):
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
    systems_df = pd.read_csv(systems_file, sep='\t', header=0, usecols=[0, 1, 4, 6])
    systems_df["model_GF"] = systems_df["model_GF"].str.split(", ").apply(lambda x: frozenset(sorted(x)))
    systems_df["system name"] = systems_df["system name"].apply(lambda x: padloc2dfinder.get(x, x))
    group_df = systems_df.groupby("system name")["model_GF"].apply(set)
    return group_df.to_dict()
