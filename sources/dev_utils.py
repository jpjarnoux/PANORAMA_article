#!/usr/bin/env python3
# coding:utf-8

# default libraries
from typing import Dict, FrozenSet, List, Set, Tuple
from collections import defaultdict, Counter

padloc2dfinder = {'cbass_type_I': 'CBASS_I', 'cbass_type_II': 'CBASS_II', 'cbass_type_IIs': 'CBASS_IIs',
                  'cbass_type_III': 'CBASS_III', 'cbass_type_IV': 'CBASS_IV',

                  'druantia_type_I': 'Druantia_I', 'druantia_type_II': 'Druantia_II',
                  'druantia_type_III': 'Druantia_III', 'druantia_type_IV': 'Druantia_IV',

                  'gabija': 'Gabija', 'kiwa': 'Kiwa', 'Lamassu_Family': 'Lamassu-Fam', 'shedu': 'Shedu',
                  'thoeris_type_I': 'Thoeris_I', 'thoeris_type_II': 'Thoeris_II', 'wadjet_type_I': 'Wadjet_I',
                  'wadjet_type_II': 'Wadjet_II', 'wadjet_type_III': 'Wadjet_III',
                  'zorya_type_I': 'Zorya_TypeI', 'zorya_type_II': 'Zorya_TypeII',

                  'cas_type_I-A': 'CAS_Class1-Subtype-I-A', 'cas_type_I-B1': 'CAS_Class1-Subtype-I-B',
                  'cas_type_I-B2': 'CAS_Class1-Subtype-I-B', 'cas_type_I-C': 'CAS_Class1-Subtype-I-C',
                  'cas_type_I-D': 'CAS_Class1-Subtype-I-D', 'cas_type_I-E': 'CAS_Class1-Subtype-I-E',
                  'cas_type_I-F1': 'CAS_Class1-Subtype-I-F', 'cas_type_I-F2': 'CAS_Class1-Subtype-I-F',
                  'cas_type_I-F3': 'CAS_Class1-Subtype-I-F', 'cas_type_I-G': 'CAS_Class1-Subtype-I-G',
                  'cas_type_II-A': 'CAS_Class2-Subtype-II-A', 'cas_type_II-B': 'CAS_Class2-Subtype-II-B',
                  'cas_type_II-C': 'CAS_Class2-Subtype-II-C', 'cas_type_III-A': 'CAS_Class1-Subtype-III-A',
                  'cas_type_III-B': 'CAS_Class1-Subtype-III-B', 'cas_type_III-C': 'CAS_Class1-Subtype-III-C',
                  'cas_type_III-D': 'CAS_Class1-Subtype-III-D', 'cas_type_III-E': 'CAS_Class1-Subtype-III-E',
                  'cas_type_III-F': 'CAS_Class1-Subtype-III-F', 'cas_type_III-G': 'CAS_Class1-Type-III',
                  'cas_type_III-H': 'CAS_Class1-Type-III', 'cas_type_IV-A': 'CAS_Class1-Subtype-IV-A',
                  'cas_type_IV-B': 'CAS_Class1-Subtype-IV-B', 'cas_type_IV-C': 'CAS_Class1-Subtype-IV-C',
                  'cas_type_IV-D': 'CAS_Class1-Type-IV', 'cas_type_IV-E': 'CAS_Class1-Type-IV',
                  'cas_type_V': 'CAS_Class2-Type-V', 'cas_type_V-K': 'CAS_Class2-Type-V-K',
                  'cas_type_VI': 'CAS_Class2-Type-VI',

                  'AVAST_type_I': 'AVAST_I', 'AVAST_type_II': 'AVAST_II', 'AVAST_type_III': 'AVAST_III',
                  'AVAST_type_IV': 'AVAST_IV', 'AVAST_type_V': 'AVAST_V',

                  'DRT_class_I': 'DRT_1', 'DRT_class_II': 'DRT_2', 'DRT_type_III': 'DRT_3',

                  'dsr1': 'Dsr_I', 'dsr2': 'Dsr_II',

                  'ApeA': 'Gao_Ape', 'GAO_19': 'Gao_Her_SIR', 'GAO_20': 'Gao_Her_DUF', 'hhe': 'Gao_Hhe',
                  'ietAS': 'Gao_Iet', 'mza': 'Gao_Mza', 'ppl': 'Gao_Ppl', 'qatABCD': 'Gao_Qat', 'TerY-P': 'Gao_TerY',
                  'tmn': 'Gao_Tmn', 'upx': 'Gao_Upx',

                  'AbiP': 'AbiP2', 'Old': 'Old_exonuclease', 'Juk': 'JukAB', 'mads': 'MADS',

                  'retron_I-A': 'Retron_I_A', 'retron_I-B': 'Retron_I_B', 'retron_I-C': 'Retron_I_C',
                  'retron_II-A': 'Retron_II', 'retron_III-A': 'Retron_III', 'retron_IV': 'Retron_IV',
                  'retron_V': 'Retron_V', 'retron_VI': 'Retron_VI', 'retron_VII-A1': 'Retron_VII_1',
                  'retron_VII-A2': 'Retron_VII_2', 'retron_VIII': 'Retron_VIII', 'retron_IX': 'Retron_IX',
                  'retron_X': 'Retron_X', 'retron_XI': 'Retron_XI', 'retron_XII': 'Retron_XII',
                  'retron_XIII': 'Retron_XIII',

                  'brex_type_I': 'BREX_I', 'brex_type_II': 'BREX_II', 'brex_type_III': 'BREX_III',
                  'brex_type_IV': 'BREX_IV', 'brex_type_V': 'BREX_V', 'brex_type_VI': 'BREX_VI',

                  'disarm_type_I': 'DISARM_1', 'disarm_type_II': 'DISARM_2',
                  'PT_DndABCDE': 'Dnd_ABCDE', 'PT_DndFGH': 'Dnd_ABCDEFGH',

                  '2TM-1TM-TIR': 'Rst_2TM_1TM_TIR', '3HP': 'Rst_3HP', 'DUF4238': 'Rst_DUF4238',
                  'gop_beta_cll': 'Rst_gop_beta_cll', 'Helicase-DUF2290': 'Rst_HelicaseDUF2290',
                  'Hydrolase-TM': 'Rst_Hydrolase-Tm', 'TIR-NLR': 'Rst_TIR-NLR',

                  'gasdermin': 'GasderMIN', 'darTG': 'DarTG',

                  'RM_type_I': 'RM_Type_I', 'RM_type_II': 'RM_Type_II', 'RM_type_IIG': 'RM_Type_IIG',
                  'RM_type_III': 'RM_Type_III', 'RM_type_IV': 'RM_Type_IV',

                  'ISG15-like': 'ISG15-like', 'Mokosh_TypeI': 'Mokosh_Type_I', 'Mokosh_TypeII': 'Mokosh_Type_II',

                  'Dynamins': 'Eleos'}

type2category = {'CAS_Class1-Subtype-I-A': 'Cas', 'CAS_Class1-Subtype-I-B': 'Cas', 'CAS_Class1-Subtype-I-C': 'Cas',
                 'CAS_Class1-Subtype-I-D': 'Cas', 'CAS_Class1-Subtype-I-E': 'Cas', 'CAS_Class1-Subtype-I-F': 'Cas',
                 'CAS_Class1-Subtype-I-G': 'Cas', 'CAS_Class1-Subtype-III-A': 'Cas', 'CAS_Class1-Subtype-III-B': 'Cas',
                 'CAS_Class1-Subtype-III-C': 'Cas', 'CAS_Class1-Subtype-III-D': 'Cas',
                 'CAS_Class1-Subtype-III-E': 'Cas',
                 'CAS_Class1-Subtype-III-F': 'Cas', 'CAS_Class1-Subtype-IV-A': 'Cas', 'CAS_Class1-Subtype-IV-B': 'Cas',
                 'CAS_Class1-Subtype-IV-C': 'Cas', 'CAS_Class1-Subtype-IV-D': 'Cas', 'CAS_Class1-Type-I': 'Cas',
                 'CAS_Class1-Type-III': 'Cas',
                 'CAS_Class1-Type-IV': 'Cas', 'CAS_Class1-Subtype-IV-E': 'Cas', 'CAS_Class2-Subtype-II-A': 'Cas',
                 'CAS_Class2-Subtype-II-B': 'Cas', 'CAS_Class2-Subtype-II-C': 'Cas', 'CAS_Class2-Subtype-V-A': 'Cas',
                 'CAS_Class2-Subtype-V-B': 'Cas', 'CAS_Class2-Subtype-V-C': 'Cas', 'CAS_Class2-Subtype-V-D': 'Cas',
                 'CAS_Class2-Subtype-V-E': 'Cas', 'CAS_Class2-Subtype-V-F': 'Cas', 'CAS_Class2-Subtype-V-G': 'Cas',
                 'CAS_Class2-Subtype-V-H': 'Cas', 'CAS_Class2-Subtype-V-I': 'Cas', 'CAS_Class2-Subtype-V-K': 'Cas',
                 'CAS_Class2-Subtype-VI-A': 'Cas', 'CAS_Class2-Subtype-VI-B': 'Cas', 'CAS_Class2-Subtype-VI-C': 'Cas',
                 'CAS_Class2-Subtype-VI-D': 'Cas', 'CAS_Class2-Type-II': 'Cas', 'CAS_Class2-Type-V': 'Cas',
                 'CAS_Class2-Type-VI': 'Cas', 'CAS_Class2-Type-V-K': 'Cas', "CAS_Class2-Subtype-VI-X": "Cas",
                 "CAS_Class2-Subtype-VI-Y": "Cas", 'CAS_Cluster': 'Cas', 'cas_adaptation': 'Cas',

                 'Abi2': 'Abi', 'AbiB': 'Abi', 'AbiC': 'Abi', 'AbiD': 'Abi', 'AbiE': 'Abi', 'AbiG': 'Abi',
                 'AbiH': 'Abi',
                 'AbiI': 'Abi', 'AbiJ': 'Abi', 'AbiL': 'Abi', 'AbiN': 'Abi', 'AbiO': 'Abi', 'AbiQ': 'Abi',
                 'AbiR': 'Abi',
                 'AbiT': 'Abi', 'AbiU': 'Abi', 'AbiV': 'Abi', 'AbiZ': 'Abi', 'AbiA_large': 'Abi', 'AbiA_small': 'Abi',
                 'AbiK': 'Abi', 'AbiP2': 'Abi', 'AbiO-Nhi_family': "Abi",

                 'argonaute_other': 'argonaute', 'argonaute_solo': 'argonaute', 'argonaute_type_SiAgo': 'argonaute',
                 'argonaute_type_I': 'argonaute', 'argonaute_type_II': 'argonaute', 'argonaute_type_III': 'argonaute',

                 'AVAST_I': 'AVAST', 'AVAST_II': 'AVAST', 'AVAST_III': 'AVAST', 'AVAST_IV': 'AVAST', 'AVAST_V': 'AVAST',

                 'Avs_I': 'AVS', 'Avs_II': 'AVS', 'Avs_III': 'AVS', 'Avs_IV': 'AVS', 'Avs_V': 'AVS',

                 'BREX': 'BREX', 'BREX_I': 'BREX', 'BREX_II': 'BREX', 'BREX_III': 'BREX', 'BREX_IV': 'BREX',
                 'BREX_V': 'BREX', 'BREX_VI': 'BREX',

                 'Butters_gp30_gp31': 'Butters', 'Butters_gp57r': 'Butters',

                 'CBASS': 'CBASS', 'CBASS_I': 'CBASS', 'CBASS_II': 'CBASS', 'CBASS_IIs': 'CBASS',
                 'CBASS_III': 'CBASS', 'CBASS_IV': 'CBASS',

                 'CARD_NLR_Endonuclease': 'CARD_NLR', 'CARD_NLR_GasderMIN': 'CARD_NLR', 'CARD_NLR_Phospho': 'CARD_NLR',
                 'CARD_NLR_Subtilase': 'CARD_NLR', 'CARD_NLR_like': 'CARD_NLR',

                 'Detocs': 'Detocs', 'Detocs_hydrolase': 'Detocs', 'Detocs_REase': 'Detocs', 'Detocs_TOPRIM': 'Detocs',

                 'DRT_1': 'DRT', 'DRT_2': 'DRT', 'DRT_3': 'DRT', 'DRT_4': 'DRT', 'DRT_5': 'DRT', 'DRT6': 'DRT',
                 'DRT7': 'DRT', 'DRT8': 'DRT', 'DRT9': 'DRT', 'DRT_class_III': 'DRT',

                 'Druantia_I': 'Druantia', 'Druantia_II': 'Druantia', 'Druantia_III': 'Druantia',

                 'Dsr_I': 'Dsr', 'Dsr_II': 'Dsr',

                 'Hachiman': 'Hachiman', 'hachiman_type_I': 'Hachiman', 'hachiman_type_II': 'Hachiman',

                 'Gao_Ape': 'Gao', 'Gao_Hhe': 'Gao', 'Gao_Iet': 'Gao', 'Gao_Mza': 'Gao', 'Gao_Ppl': 'Gao',
                 'Gao_TerY': 'Gao', 'Gao_Tmn': 'Gao', 'Gao_Upx': 'Gao', 'Gao_RL': 'Gao', 'Gao_Her_DUF': 'Gao',
                 'Gao_Her_SIR': 'Gao', 'Gao_Qat': 'Gao', 'GAO_19': 'Gao', 'GAO_20': 'Gao', 'GAO_29': 'Gao',

                 'HEC-01': 'HEC', 'HEC-02': 'HEC', 'HEC-03': 'HEC', 'HEC-04': 'HEC', 'HEC-05': 'HEC', 'HEC-06': 'HEC',
                 'HEC-07': 'HEC', 'HEC-08': 'HEC', 'HEC-09': 'HEC',

                 'PD-T4-1': 'PD', 'PD-T4-2': 'PD', 'PD-T4-3': 'PD', 'PD-T4-4': 'PD', 'PD-T4-5': 'PD', 'PD-T4-6': 'PD',
                 'PD-T4-7': 'PD', 'PD-T4-8': 'PD', 'PD-T4-9': 'PD', 'PD-T4-10': 'PD', 'PD-T7-1': 'PD', 'PD-T7-2': 'PD',
                 'PD-T7-3': 'PD', 'PD-T7-4': 'PD', 'PD-T7-5': 'PD', 'PD-T7-1or5': 'PD', 'PD-Lambda-1': 'PD',
                 'PD-Lambda-2': 'PD', 'PD-Lambda-3': 'PD', 'PD-Lambda-4': 'PD', 'PD-Lambda-5': 'PD',
                 'PD-Lambda-6': 'PD',

                 'PDC-M01': 'PDC', 'PDC-M02': 'PDC', 'PDC-M03': 'PDC', 'PDC-M04': 'PDC', 'PDC-M05': 'PDC',
                 'PDC-M06': 'PDC', 'PDC-M07': 'PDC', 'PDC-M08': 'PDC', 'PDC-M09': 'PDC', 'PDC-M10': 'PDC',
                 'PDC-M11': 'PDC', 'PDC-M12': 'PDC', 'PDC-M13': 'PDC', 'PDC-M14': 'PDC', 'PDC-M15': 'PDC',
                 'PDC-M16': 'PDC', 'PDC-M17': 'PDC', 'PDC-M18': 'PDC', 'PDC-M19': 'PDC', 'PDC-M20': 'PDC',
                 'PDC-M21': 'PDC', 'PDC-M22': 'PDC', 'PDC-M23': 'PDC', 'PDC-M24': 'PDC', 'PDC-M25': 'PDC',
                 'PDC-M26': 'PDC', 'PDC-M27': 'PDC', 'PDC-M28': 'PDC', 'PDC-M29': 'PDC', 'PDC-M30': 'PDC',
                 'PDC-M31': 'PDC', 'PDC-M32': 'PDC', 'PDC-M33': 'PDC', 'PDC-M34': 'PDC', 'PDC-M35': 'PDC',
                 'PDC-M36': 'PDC', 'PDC-M37': 'PDC', 'PDC-M38': 'PDC', 'PDC-M39': 'PDC', 'PDC-M40': 'PDC',
                 'PDC-M41': 'PDC', 'PDC-M42': 'PDC', 'PDC-M43': 'PDC', 'PDC-M44': 'PDC', 'PDC-M45': 'PDC',
                 'PDC-M46': 'PDC', 'PDC-M47': 'PDC', 'PDC-M48': 'PDC', 'PDC-M49': 'PDC', 'PDC-M50': 'PDC',
                 'PDC-M51': 'PDC', 'PDC-M52': 'PDC', 'PDC-M53': 'PDC', 'PDC-M54': 'PDC', 'PDC-M55': 'PDC',
                 'PDC-M56': 'PDC', 'PDC-M57': 'PDC', 'PDC-M58': 'PDC', 'PDC-M59': 'PDC', 'PDC-M60': 'PDC',
                 'PDC-M61': 'PDC', 'PDC-M62': 'PDC', 'PDC-M63': 'PDC', 'PDC-M64': 'PDC', 'PDC-M65': 'PDC',
                 'PDC-M66': 'PDC', 'PDC-M67': 'PDC', 'PDC-M68': 'PDC', 'PDC-M69': 'PDC', 'PDC-M70': 'PDC',
                 'PDC-M71': 'PDC', 'PDC-M72': 'PDC', 'PDC-S01': 'PDC', 'PDC-S02': 'PDC', 'PDC-S03': 'PDC',
                 'PDC-S04': 'PDC', 'PDC-S05': 'PDC', 'PDC-S06': 'PDC', 'PDC-S07': 'PDC', 'PDC-S08': 'PDC',
                 'PDC-S09': 'PDC', 'PDC-S10': 'PDC', 'PDC-S11': 'PDC', 'PDC-S12': 'PDC', 'PDC-S13': 'PDC',
                 'PDC-S14': 'PDC', 'PDC-S15': 'PDC', 'PDC-S16': 'PDC', 'PDC-S17': 'PDC', 'PDC-S18': 'PDC',
                 'PDC-S19': 'PDC', 'PDC-S20': 'PDC', 'PDC-S21': 'PDC', 'PDC-S22': 'PDC', 'PDC-S23': 'PDC',
                 'PDC-S24': 'PDC', 'PDC-S25': 'PDC', 'PDC-S26': 'PDC', 'PDC-S27': 'PDC', 'PDC-S28': 'PDC',
                 'PDC-S29': 'PDC', 'PDC-S30': 'PDC', 'PDC-S31': 'PDC', 'PDC-S32': 'PDC', 'PDC-S33': 'PDC',
                 'PDC-S34': 'PDC', 'PDC-S35': 'PDC', 'PDC-S36': 'PDC', 'PDC-S37': 'PDC', 'PDC-S38': 'PDC',
                 'PDC-S39': 'PDC', 'PDC-S40': 'PDC', 'PDC-S41': 'PDC', 'PDC-S42': 'PDC', 'PDC-S43': 'PDC',
                 'PDC-S44': 'PDC', 'PDC-S45': 'PDC', 'PDC-S46': 'PDC', 'PDC-S47': 'PDC', 'PDC-S48': 'PDC',
                 'PDC-S49': 'PDC', 'PDC-S50': 'PDC', 'PDC-S51': 'PDC', 'PDC-S52': 'PDC', 'PDC-S53': 'PDC',
                 'PDC-S54': 'PDC', 'PDC-S55': 'PDC', 'PDC-S56': 'PDC', 'PDC-S57': 'PDC', 'PDC-S58': 'PDC',
                 'PDC-S59': 'PDC', 'PDC-S60': 'PDC', 'PDC-S61': 'PDC', 'PDC-S62': 'PDC', 'PDC-S63': 'PDC',
                 'PDC-S64': 'PDC', 'PDC-S65': 'PDC', 'PDC-S66': 'PDC', 'PDC-S67': 'PDC', 'PDC-S68': 'PDC',
                 'PDC-S69': 'PDC', 'PDC-S70': 'PDC', 'PDC-S71': 'PDC', 'PDC-S72': 'PDC', 'PDC-S73': 'PDC',

                 'pycsar_other': 'pyscar', 'pycsar_effector': 'pyscar', 'pycsar_unknown': 'pyscar',

                 'PifA': 'PifA', 'MADS': 'MADS',

                 'Retron_I_A': 'Retron', 'Retron_I_B': 'Retron', 'Retron_I_C': 'Retron', 'Retron_II': 'Retron',
                 'Retron_III': 'Retron', 'Retron_IV': 'Retron', 'Retron_V': 'Retron', 'Retron_VI': 'Retron',
                 'Retron_VII_1': 'Retron', 'Retron_VII_2': 'Retron', 'Retron_VIII': 'Retron', 'Retron_IX': 'Retron',
                 'Retron_X': 'Retron', 'Retron_XI': 'Retron', 'Retron_XII': 'Retron', 'Retron_XIII': 'Retron',

                 'RM_Type_I': 'RM', 'RM_Type_II': 'RM', 'RM_Type_IIG': 'RM', 'RM_Type_III': 'RM', 'RM_Type_IV': 'RM',
                 'RM_type_HNH': 'RM',

                 'radar_I': 'RADAR', 'radar_II': 'RADAR', 'RADAR': 'RADAR',

                 'Rst_3HP': 'Rst', 'Rst_DUF4238': 'Rst', 'Rst_HelicaseDUF2290': 'Rst', 'Rst_Hydrolase-Tm': 'Rst',
                 'Rst_RT-Tm': 'Rst', 'Rst_TIR-NLR': 'Rst', 'Rst_2TM_1TM_TIR': 'Rst', 'Rst_gop_beta_cll': 'Rst',
                 'PARIS_I': 'PARIS', 'PARIS_II': 'PARIS', 'PARIS_II_merge': 'PARIS',
                 'PARIS_I_merge': 'PARIS', 'Paris_fused': 'PARIS', 'Paris': 'PARIS',

                 'Wadjet': 'Wadjet', 'Wadjet_I': 'Wadjet', 'Wadjet_II': 'Wadjet', 'Wadjet_III': 'Wadjet',

                 'BstA': 'BstA', 'DISARM_1': 'DISARM', 'DISARM_2': 'DISARM', 'Gabija': 'Gabija',

                 'PT_SspABCD': 'PT', 'PT_PbeABCD': 'PT', 'PT_SspFGH': 'PT', 'PT_SspE': 'PT',

                 'septu_other': 'septu', 'septu_type_I': 'septu',

                 'hma': 'hma', 'Hna': 'Hna', 'hhe': 'hhe', 'JukAB': 'Juk', 'Eleos': 'Eleos', 'MazEF': 'MazEF',

                 'pAgo': 'pAgo', 'pAgo_LongA': 'pAgo', 'pAgo_LongB': 'pAgo', 'pAgo_S1A': 'pAgo', 'pAgo_S1B': 'pAgo',
                 'pAgo_S2B': 'pAgo', 'pAgo_SPARTA': 'pAgo',

                 'Lamassu-Fam': 'Lamassu', 'Lamassu-Amidase': 'Lamassu', 'Lamassu-Cap4_nuclease': 'Lamassu',
                 'Lamassu-FMO': 'Lamassu', 'Lamassu-Hydrolase': 'Lamassu', 'Lamassu-Hydrolase_Protease': 'Lamassu',
                 'Lamassu-Hypothetical': 'Lamassu', 'Lamassu-Lipase': 'Lamassu', 'Lamassu-Mrr': 'Lamassu',
                 'Lamassu-PDDEXK': 'Lamassu', 'Lamassu-Protease': 'Lamassu', 'Lamassu-Sir2': 'Lamassu',

                 'Mokosh_Type_I': 'Mokosh', 'Mokosh_Type_I_A': 'Mokosh', 'Mokosh_Type_I_B': 'Mokosh',
                 'Mokosh_Type_I_C': 'Mokosh', 'Mokosh_Type_I_D': 'Mokosh', 'Mokosh_Type_I_E': 'Mokosh',
                 'Mokosh_Type_II': 'Mokosh', 'Mokosh_TypeII': 'Mokosh',

                 'FS_Sma': 'FS', 'FS_HsdR_like': 'FS', 'FS_HP_SDH_sah': 'FS', 'FS_HP': 'FS', 'FS_HEPN_TM': 'FS',
                 'FS_GIY_YIG': 'FS',

                 'GAPS6': 'GAP', 'GAPS2': 'GAP', 'GAPS1': 'GAP', 'GAPS4': 'GAP',

                 'GasderMIN': 'GasderMIN', 'Kiwa': 'Kiwa', 'Lit': 'Lit', 'Nhi': 'Nhi',
                 'NixI': 'NixI', 'Pif': 'Pif', 'RexAB': 'RexAB', 'RloC': 'RloC',
                 'Shedu': 'Shedu', 'Stk2': 'Stk2', 'Viperin': 'Viperin', 'dCTPdeaminase': 'dCTP',
                 'dGTPase': 'dGTPase', 'DarTG': 'DarTG', 'MqsRAC': 'MqsRAC',
                 'NLR_like_bNACHT01': 'NLR', 'NLR_like_bNACHT09': 'NLR', 'Old_exonuclease': 'Old_exonuclease',
                 'PsyrTA': 'PsyrTA', 'Septu': 'Septu', 'ShosTA': 'ShosTA', 'Azaca': 'Azaca', 'CapRel': 'CapRel',
                 'DdmDE': 'DdmDE', 'SpbK': 'SpbK', 'Thoeris_I': 'Thoeris', 'Thoeris_II': 'Thoeris',
                 'gp29_gp30': 'gp29_gp30', 'Dnd_ABCDE': 'Dnd', 'Dnd_ABCDEFGH': 'Dnd', 'Dodola': 'Dodola',
                 'Dpd': 'Dpd', 'ISG15-like': 'ISG15-like', 'Menshen': 'Menshen',
                 'Mok_Hok_Sok': 'Mok_Hok_Sok',
                 'RosmerTA': 'RosmerTA', 'Shango': 'Shango', 'SspBCDE': 'SspBCDE', 'Zorya_TypeI': 'Zorya',
                 'Zorya_TypeII': 'Zorya', 'Aditi': 'Aditi', 'Borvo': 'Borvo', 'Bunzi': 'Bunzi', 'Dazbog': 'Dazbog',
                 'Dynamins': 'Dynamins', 'Olokun': 'Olokun', 'PfiAT': 'PfiAT', 'Pycsar': 'Pycsar', 'RnlAB': 'RnlAB',
                 'SEFIR': 'SEFIR', 'SanaTA': 'SanaTA', 'SoFic': 'SoFIC', 'Tiamat': 'Tiamat', 'Uzume': 'Uzume',
                 'PrrC': 'PrrC',

                 'VSPR': 'VSPR'}


def vote_partitions(partitions: List[str], threshold: float = 0.7):
    counter = Counter(partitions)
    max_counter = max(counter.values())
    if max_counter / len(partitions) >= threshold:
        return counter.most_common(1)[0][0]
    else:
        return "undecided"


def cluster_low_sys(group_counter: Dict[str, int], threshold: float = 0.85) -> Tuple[Set[str], Set[str]]:
    # Calculer la somme des individus dans les groupes les plus grands
    count_sum = 0
    main_group = set()
    count_total = sum(group_counter.values())
    for system, count in sorted(group_counter.items(), key=lambda x: x[1], reverse=True):
        count_sum += count
        main_group.add(system)
        if count_sum / count_total >= threshold:
            break

    # Regrouper les individus restants dans le groupe "Other"
    others_group = {system for system in group_counter.keys() if system not in main_group}
    return main_group, others_group


def get_tool_systems(tool: Dict[str, Dict[str, List[Set[str]]]]) -> Dict[str, Set[FrozenSet[str]]]:
    """
    Extract unique systems from the provided tool dictionary and update the mapper with new systems.

    Args:
        tool (Dict[str, Dict[str, List[Set[str]]]]): Nested dictionary of systems by organization and type.

    Returns:
        Dict[str, Set[FrozenSet[str]]]: Dict of Set of unique systems as frozensets for each system type.
    """
    tool_sys = defaultdict(set)
    for systems_by_org in tool.values():
        for system_type, systems_by_type in systems_by_org.items():
            for system in sorted(systems_by_type, key=len, reverse=True):
                frozen_system = frozenset(sorted(system))
                if frozen_system not in tool_sys[system_type]:
                    tool_sys[system_type].add(frozen_system)
    return tool_sys


def similar(sys1: Set[str], sys2: Set[str], threshold: float = 0.5) -> bool:
    inter = sys1 & sys2
    minimum = min(len(sys1), len(sys2))
    similarity = len(inter) / minimum
    return similarity >= threshold


def merge_results(
        dict1: Dict[str, Set[FrozenSet[str]]],
        dict2: Dict[str, Set[FrozenSet[str]]],
        threshold: float = 0.5
) -> Dict[str, Set[FrozenSet[str]]]:
    merged_dict = {}

    # Combine keys from both dictionaries
    for key in set(dict1.keys()).union(dict2.keys()):
        all_frozensets = list(dict1.get(key, set())) + list(dict2.get(key, set()))
        all_frozensets = list(sorted(all_frozensets, key=len, reverse=True))
        merged_set = []

        while all_frozensets:
            to_merge = {all_frozensets.pop(0)}

            # Check for similar sets within all remaining frozensets
            i = 0
            while i < len(all_frozensets):
                fs = all_frozensets[i]
                if any(similar(current_fs, fs, threshold) for current_fs in to_merge):
                    to_merge.add(fs)
                    all_frozensets.pop(i)  # Remove fs since it will be merged
                else:
                    i += 1

            # Merge all similar frozensets found
            merged_fs = frozenset().union(*to_merge)
            merged_set.append(merged_fs)

        # Add merged frozensets to the result dictionary for the current key
        merged_dict[key] = set(merged_set)

    return merged_dict
