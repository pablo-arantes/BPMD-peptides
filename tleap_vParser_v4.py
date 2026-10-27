#!/usr/bin/env python3
# ============================================================
# tleap_parse.py
# Topology generation with SSBOND control and ionization overrides
#
# Usage:
#   python tleap_parse.py --ff ff14SB --ph 7.0 --ssbond_complex no
#   python tleap_parse.py --ff ff19SB --ph 6.5 --ssbond_complex yes \
#                         --ion_aminoacid_list ionization_amioacid.txt
# ============================================================

import os
import subprocess
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pymol import cmd
from openmm.app.pdbfile import PDBFile
from pdbfixer import PDBFixer
from openbabel import pybel
from biopandas.pdb import PandasPdb

# ======================== FORCE FIELD CONFIG ========================
FF_CONFIG = {
    "ff14SB": {
        "leaprc": "leaprc.protein.ff14SB",
        "water":  "leaprc.water.tip3p",
        "water_box": "TIP3PBOX",
    },
    "ff19SB": {
        "leaprc": "leaprc.protein.ff19SB",
        "water":  "leaprc.water.opc",
        "water_box": "OPCBOX",
    },
}

# Amino acid ionization families: residues that share the same base
# and differ only in protonation / tautomeric state.
# Used to validate ionization overrides -- prevents renaming a VAL to GLH.
IONIZATION_FAMILIES = {
    'GLH': {'GLU', 'GLH'},
    'GLU': {'GLU', 'GLH'},
    'ASH': {'ASP', 'ASH'},
    'ASP': {'ASP', 'ASH'},
    'HID': {'HIS', 'HID', 'HIE', 'HIP'},
    'HIE': {'HIS', 'HID', 'HIE', 'HIP'},
    'HIP': {'HIS', 'HID', 'HIE', 'HIP'},
    'HIS': {'HIS', 'HID', 'HIE', 'HIP'},
    'CYS': {'CYS', 'CYX'},
    'CYX': {'CYS', 'CYX'},
    'LYN': {'LYS', 'LYN'},
    'LYS': {'LYS', 'LYN'},
}


# ======================== ARGUMENT PARSING ========================
def parse_topology_arguments():
    parser = argparse.ArgumentParser(
        description="Topology generation with SSBOND and ionization control"
    )
    parser.add_argument(
        "--ff", choices=FF_CONFIG.keys(), default="ff14SB",
        help="Protein force field (ff14SB -> TIP3P, ff19SB -> OPC)"
    )
    parser.add_argument(
        "--size_box", type=float, default=10.0,
        help="Solvation box size (Angstrom)"
    )
    parser.add_argument(
        "--concentration", type=float, default=0.15,
        help="Salt concentration (M)"
    )
    parser.add_argument(
        "--ph", type=float, default=7.0,
        help="System pH"
    )
    parser.add_argument(
        "--ssbond_complex",
        choices=["yes", "no"], default="no",
        help=(
            "Allow inter-chain disulfide bonds. "
            "'yes' = include inter-chain SSBONDs in tleap; "
            "'no'  = intra-chain only, inter-chain CYX reverted to CYS."
        )
    )
    parser.add_argument(
        "--ion_aminoacid_list",
        type=str, default=None,
        help=(
            "Path to tab-separated file with user-preferred ionization "
            "states. Residue numbers refer to the ORIGINAL PDB numbering. "
            "Format: RESNAME<TAB>RESID[<TAB>CHAIN]. "
            "Overrides pdb2pqr pKa-based assignments."
        )
    )

    args = parser.parse_args()

    if args.concentration < 0:
        raise ValueError("Salt concentration must be non-negative")
    if not (0.0 <= args.ph <= 14.0):
        raise ValueError("pH must be between 0 and 14")

    # Convert ssbond_complex from string to bool
    args.ssbond_complex = (args.ssbond_complex.lower() == "yes")

    return args


# ======================== PARSING FUNCTIONS ========================
def parse_ssbond_records(pdb_file):
    """Parse SSBOND records from PDB file header.

    Robustly finds CYS tokens to handle varying PDB formats
    (with/without serial numbers, different spacing).

    Returns
    -------
    list of dict : {'chain1', 'resid1', 'chain2', 'resid2', 'is_interchain'}
    """
    ssbonds = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('SSBOND'):
                parts = line.split()
                cys_idx = [i for i, p in enumerate(parts) if p == 'CYS']
                if len(cys_idx) >= 2:
                    chain1 = parts[cys_idx[0] + 1]
                    resid1 = int(parts[cys_idx[0] + 2])
                    chain2 = parts[cys_idx[1] + 1]
                    resid2 = int(parts[cys_idx[1] + 2])
                    ssbonds.append({
                        'chain1': chain1, 'resid1': resid1,
                        'chain2': chain2, 'resid2': resid2,
                        'is_interchain': chain1 != chain2,
                    })
            # Stop once we reach coordinate records
            if line.startswith(('ATOM', 'HETATM')):
                break
    return ssbonds


def parse_ionization_preferences(ion_file):
    """Parse user ionization preference file.

    Residue numbers refer to the ORIGINAL PDB numbering.
    The code will automatically translate them to the processed
    PDB numbering (which may differ when chains are merged).

    Format (tab-separated, one per line):
        RESNAME<TAB>RESID           -> applies to all chains
        RESNAME<TAB>RESID<TAB>CHAIN -> chain-specific
    Lines starting with '#' are comments.

    Returns
    -------
    list of dict : {'resname', 'resid', 'chain' (optional)}
    """
    preferences = []
    if ion_file is None or not os.path.exists(ion_file):
        return preferences
    with open(ion_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            pref = {'resname': parts[0].strip(), 'resid': int(parts[1].strip())}
            if len(parts) > 2:
                pref['chain'] = parts[2].strip()
            preferences.append(pref)
    return preferences


def build_pdb_residue_mapping(original_pdb, processed_pdb):
    """Build a mapping from original PDB (chain, resid) -> processed PDB (chain, resid).

    Matches residues by their sequential position in the file.
    Handles renumbering that occurs when pdb2pqr merges chains
    or renumbers residues sequentially (e.g. chain A: 1-30,
    chain B: 1-20 -> processed: 1-50).

    Returns
    -------
    dict : {(orig_chain, orig_resid): (proc_chain, proc_resid)}
    """
    def get_ordered_residues(pdb_file):
        ppdb = PandasPdb().read_pdb(pdb_file)
        atoms = ppdb.df['ATOM']
        residues = atoms[['chain_id', 'residue_number']].drop_duplicates()
        return [(row['chain_id'], int(row['residue_number']))
                for _, row in residues.iterrows()]

    orig_residues = get_ordered_residues(original_pdb)
    proc_residues = get_ordered_residues(processed_pdb)

    mapping = {}
    for orig, proc in zip(orig_residues, proc_residues):
        mapping[orig] = proc

    if len(orig_residues) != len(proc_residues):
        print(f"    WARNING: residue count mismatch -- original "
              f"{len(orig_residues)}, processed {len(proc_residues)}. "
              f"Mapping may be incomplete.")

    return mapping


def build_tleap_residue_mapping(pdb_file):
    """Map (chain_id, residue_number) -> sequential tleap residue index (1-based).

    tleap numbers residues sequentially across all chains, so PDB
    residue numbers must be translated for multi-chain systems.
    """
    ppdb = PandasPdb().read_pdb(pdb_file)
    atoms = ppdb.df['ATOM']
    residues = atoms[['chain_id', 'residue_number']].drop_duplicates()
    mapping = {}
    for idx, (_, row) in enumerate(residues.iterrows(), 1):
        mapping[(row['chain_id'], int(row['residue_number']))] = idx
    return mapping


# ======================== RESIDUE MODIFICATIONS ========================
def apply_residue_modifications(pdb_file, ssbonds, ssbond_complex,
                                ionization_prefs, orig_to_proc_mapping):
    """Modify residue names in the pdb2pqr output PDB.

    Uses orig_to_proc_mapping to translate original PDB (chain, resid)
    to processed PDB (chain, resid), handling renumbering across chains.

    1. If ssbond_complex=False: revert CYX -> CYS for inter-chain SSBONDs.
    2. Apply ionization overrides, validated per chain against
       IONIZATION_FAMILIES to avoid renaming incompatible residues.
    """
    ppdb = PandasPdb().read_pdb(pdb_file)

    # --- 1. Inter-chain SSBOND: CYX -> CYS ---
    n_reverted = 0
    if not ssbond_complex:
        for ss in ssbonds:
            if ss['is_interchain']:
                for orig_chain, orig_resid in [(ss['chain1'], ss['resid1']),
                                               (ss['chain2'], ss['resid2'])]:
                    # Translate to processed PDB numbering
                    proc_key = orig_to_proc_mapping.get(
                        (orig_chain, orig_resid), (orig_chain, orig_resid)
                    )
                    proc_chain, proc_resid = proc_key

                    mask = (
                        (ppdb.df['ATOM']['chain_id'] == proc_chain) &
                        (ppdb.df['ATOM']['residue_number'] == proc_resid) &
                        (ppdb.df['ATOM']['residue_name'] == 'CYX')
                    )
                    if mask.any():
                        ppdb.df['ATOM'].loc[mask, 'residue_name'] = 'CYS'
                        n_reverted += 1

    # --- 2. Ionization overrides (validated, mapped per chain) ---
    n_overrides = 0
    for pref in ionization_prefs:
        target_name = pref['resname']
        compatible  = IONIZATION_FAMILIES.get(target_name)

        # Collect original-PDB keys for this preference
        if 'chain' in pref:
            orig_keys = [(pref['chain'], pref['resid'])]
        else:
            # Find all chains in original PDB that have this residue number
            orig_keys = [k for k in orig_to_proc_mapping if k[1] == pref['resid']]

        if not orig_keys:
            print(f"    WARNING: residue {pref['resid']} not found in "
                  f"original PDB mapping -- skipping")
            continue

        for orig_key in orig_keys:
            # Translate to processed PDB numbering
            proc_key = orig_to_proc_mapping.get(orig_key)
            if proc_key is None:
                print(f"    WARNING: original {orig_key} has no mapping "
                      f"to processed PDB -- skipping")
                continue
            proc_chain, proc_resid = proc_key

            mask = (
                (ppdb.df['ATOM']['chain_id'] == proc_chain) &
                (ppdb.df['ATOM']['residue_number'] == proc_resid)
            )
            if not mask.any():
                print(f"    WARNING: mapped residue ({proc_chain}, "
                      f"{proc_resid}) not found in processed PDB -- skipping")
                continue

            current_name = ppdb.df['ATOM'].loc[mask, 'residue_name'].iloc[0]

            # Validate: only rename if current residue belongs to the
            # same ionization family (e.g. GLU<->GLH, ASP<->ASH).
            if compatible and current_name not in compatible:
                print(f"    WARNING: original {orig_key} mapped to "
                      f"({proc_chain}, {proc_resid}) is {current_name}, "
                      f"not compatible with {target_name} "
                      f"(expected {compatible}) -- skipping")
                continue

            ppdb.df['ATOM'].loc[mask, 'residue_name'] = target_name
            print(f"    Ionization override: original chain "
                  f"{orig_key[0]} resid {orig_key[1]} -> processed "
                  f"({proc_chain}, {proc_resid}) {current_name} -> "
                  f"{target_name}")
            n_overrides += 1

    # Save modified PDB (overwrite)
    ppdb.to_pdb(path=pdb_file, records=['ATOM', 'HETATM'],
                gz=False, append_newline=True)
    print(f"    Residue modifications: {n_reverted} CYX->CYS reverted, "
          f"{n_overrides} ionization overrides applied")


# ======================== PDB PREPROCESSING ========================
def preprocess_pdb(pdb_file, work_dir, ph, ssbonds, ssbond_complex, ionization_prefs):
    """Full PDB preprocessing pipeline:
    1. pdb2pqr  ->  protonation-state assignment
    2. Build residue mapping (original -> pdb2pqr numbering)
    3. Residue modifications  ->  CYX->CYS + ionization overrides
    4. PyMOL    ->  structural cleanup
    5. BioPandas ->  remove water, OXT, hydrogens
    """
    name = os.path.splitext(os.path.basename(pdb_file))[0]
    pdb_output_dir = os.path.join(work_dir, name)
    os.makedirs(pdb_output_dir, exist_ok=True)

    pdb2pqr_output = os.path.join(pdb_output_dir, f"{name}_pka")

    # Step 1: pdb2pqr
    print(f"  Running pdb2pqr (pH={ph}) ...")
    subprocess.run(
        f"pdb2pqr {pdb_file} --ff AMBER --ffout AMBER --with-ph {ph} "
        f"--pdb-output {pdb2pqr_output}.pdb {pdb2pqr_output}.pqr",
        shell=True, check=True
    )

    # Step 2: Build mapping original PDB -> pdb2pqr numbering
    print(f"  Building residue mapping (original -> pdb2pqr) ...")
    orig_to_proc_mapping = build_pdb_residue_mapping(
        pdb_file, f"{pdb2pqr_output}.pdb"
    )

    # Step 3: Apply residue modifications
    print(f"  Applying residue modifications ...")
    apply_residue_modifications(
        f"{pdb2pqr_output}.pdb", ssbonds, ssbond_complex,
        ionization_prefs, orig_to_proc_mapping
    )

    # Step 4: PyMOL cleanup
    print(f"  PyMOL cleanup ...")
    cmd.load(f"{pdb2pqr_output}.pdb", name)
    cmd.set("retain_order", "on")
    pymol_output = os.path.join(pdb_output_dir, f"{name}_pymol.pdb")
    cmd.save(pymol_output, name, format="pdb")
    cmd.delete(name)

    # Step 5: BioPandas cleanup (remove OXT, H, water)
    print(f"  BioPandas cleanup ...")
    edited_pdb_path = os.path.join(pdb_output_dir, f"{name}_ed.pdb")
    ppdb = PandasPdb().read_pdb(pymol_output)
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] != 'OXT']
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    ppdb.df['HETATM'] = ppdb.df['HETATM'][ppdb.df['HETATM']['residue_name'] != 'HOH']
    ppdb.to_pdb(path=edited_pdb_path, records=['ATOM', 'HETATM'],
                gz=False, append_newline=True)

    return edited_pdb_path


# ======================== TLEAP FUNCTIONS ========================
def get_ssbond_tleap_lines(ssbonds, ssbond_complex, residue_mapping):
    """Generate tleap 'bond' commands for disulfide bonds.

    - Intra-chain SSBONDs are ALWAYS included.
    - Inter-chain SSBONDs are included ONLY if ssbond_complex=True.
    - Uses residue_mapping to translate (chain, PDB_resid) -> tleap index.
    """
    bond_lines = []
    for ss in ssbonds:
        if ss['is_interchain'] and not ssbond_complex:
            continue
        key1 = (ss['chain1'], ss['resid1'])
        key2 = (ss['chain2'], ss['resid2'])
        idx1 = residue_mapping.get(key1)
        idx2 = residue_mapping.get(key2)
        if idx1 is not None and idx2 is not None:
            bond_lines.append(f"bond SYS.{idx1}.SG SYS.{idx2}.SG")
        else:
            missing = [k for k, v in [(key1, idx1), (key2, idx2)] if v is None]
            print(f"  WARNING: SSBOND residue(s) {missing} not found in "
                  f"final PDB -- skipping bond")
    return bond_lines


def write_tleap_input_no_ions(ff, water, water_box, size_box,
                              pdb_file, output_prefix, ssbond_lines):
    """Write tleap input without ions (used for volume estimation)."""
    tleap_file = f"{output_prefix}_no_ions_tleap.in"
    with open(tleap_file, "w") as f:
        f.write(f"source {ff}\n")
        f.write("source leaprc.DNA.OL15\n")
        f.write("source leaprc.RNA.OL3\n")
        f.write("source leaprc.GLYCAM_06j-1\n")
        f.write("source leaprc.gaff2\n")
        f.write(f"source {water}\n")
        f.write(f"SYS = loadpdb {pdb_file}\n")
        for line in ssbond_lines:
            f.write(f"{line}\n")
        f.write("alignaxes SYS\n")
        f.write(f"solvateOct SYS {water_box} {size_box}\n")
        f.write(f"saveamberparm SYS {output_prefix}_no_ions.top "
                f"{output_prefix}_no_ions.crd\n")
        f.write(f"savepdb SYS {output_prefix}_no_ions.pdb\n")
        f.write("quit\n")
    return tleap_file


def write_tleap_input_with_ions(ff, water, water_box, size_box,
                                pdb_file, output_prefix, num_ions,
                                ssbond_lines):
    """Write tleap input with neutralization + ion concentration."""
    tleap_file = f"{output_prefix}_with_ions_tleap.in"
    with open(tleap_file, "w") as f:
        f.write(f"source {ff}\n")
        f.write("source leaprc.DNA.OL15\n")
        f.write("source leaprc.RNA.OL3\n")
        f.write("source leaprc.GLYCAM_06j-1\n")
        f.write("source leaprc.gaff2\n")
        f.write(f"source {water}\n")
        f.write(f"SYS = loadpdb {pdb_file}\n")
        for line in ssbond_lines:
            f.write(f"{line}\n")
        f.write("alignaxes SYS\n")
        f.write("check SYS\n")
        f.write("charge SYS\n")
        f.write("addions SYS Na+ 0\n")
        f.write("addions SYS Cl- 0\n")
        f.write("check SYS\n")
        f.write("charge SYS\n")
        f.write(f"solvateOct SYS {water_box} {size_box}\n")
        f.write(f"addIonsRand SYS Na+ {num_ions} Cl- {num_ions}\n")
        f.write(f"saveamberparm SYS {output_prefix}_with_ions.top "
                f"{output_prefix}_with_ions.crd\n")
        f.write(f"savepdb SYS {output_prefix}_with_ions.pdb\n")
        f.write("quit\n")
    return tleap_file


def calculate_num_ions(volume, base_concentration, target_concentration):
    """Calculate number of ions from box volume and target salt concentration."""
    vol_liters = volume * 1e-27
    atom_liters = 9.03e22
    return int(vol_liters * (target_concentration / base_concentration)
               * atom_liters)


def run_tleap(tleap_input, tleap_log):
    """Execute tleap with given input file."""
    try:
        subprocess.run(f"tleap -f {tleap_input} > {tleap_log}",
                       shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR running tleap: {e}")
        return False
    return True


# ======================== MAIN PIPELINE ========================
def process_pdb(pdb_file, output_prefix, ff, water, water_box, size_box,
                concentration, ph, ssbond_complex, ion_aminoacid_list, work_dir):
    """Full pipeline: parse -> preprocess -> generate tleap -> run tleap."""
    print(f"{'='*60}")
    print(f"Processing: {pdb_file}")
    print(f"  ssbond_complex = {ssbond_complex}")
    print(f"  ion_aminoacid_list = {ion_aminoacid_list}")
    print(f"{'='*60}")

    # 1. Parse SSBOND records from the ORIGINAL PDB
    ssbonds = parse_ssbond_records(pdb_file)
    intra = sum(1 for s in ssbonds if not s['is_interchain'])
    inter = sum(1 for s in ssbonds if s['is_interchain'])
    print(f"  SSBOND records found: {len(ssbonds)} total "
          f"({intra} intra-chain, {inter} inter-chain)")
    for ss in ssbonds:
        tag = "INTER" if ss['is_interchain'] else "intra"
        print(f"    [{tag}] {ss['chain1']}-{ss['resid1']} <-> "
              f"{ss['chain2']}-{ss['resid2']}")

    if not ssbond_complex and inter > 0:
        print(f"  -> ssbond_complex=False: {inter} inter-chain bond(s) "
              f"will be excluded, CYX->CYS applied")

    # 2. Parse ionization preferences
    ionization_prefs = parse_ionization_preferences(ion_aminoacid_list)
    if ionization_prefs:
        print(f"  Ionization overrides requested: {len(ionization_prefs)}")
        for p in ionization_prefs:
            chain_info = f" chain {p['chain']}" if 'chain' in p else ""
            print(f"    resid {p['resid']}{chain_info} -> {p['resname']}")

    # 3. Preprocess PDB (pdb2pqr + mapping + modifications + cleanup)
    edited_pdb = preprocess_pdb(
        pdb_file, work_dir, ph, ssbonds, ssbond_complex, ionization_prefs
    )
    print(f"  Final cleaned PDB: {edited_pdb}")

    # 4. Build tleap residue mapping from the final cleaned PDB
    residue_mapping = build_tleap_residue_mapping(edited_pdb)
    print(f"  Residue mapping built: {len(residue_mapping)} residues")

    # 5. Generate SSBOND tleap bond lines
    ssbond_lines = get_ssbond_tleap_lines(
        ssbonds, ssbond_complex, residue_mapping
    )
    if ssbond_lines:
        print(f"  tleap bond lines ({len(ssbond_lines)}):")
        for bl in ssbond_lines:
            print(f"    {bl}")
    else:
        print(f"  No SSBOND bond lines to write.")

    # 6. tleap without ions (to estimate box volume)
    tleap_no_ions = write_tleap_input_no_ions(
        ff, water, water_box, size_box, edited_pdb, output_prefix,
        ssbond_lines
    )
    log_no_ions = f"{output_prefix}_no_ions.log"
    if not run_tleap(tleap_no_ions, log_no_ions):
        return False

    # 7. Extract volume from tleap log
    volume = None
    with open(log_no_ions, 'r') as log:
        for line in log:
            if "Volume:" in line:
                volume = float(line.split()[1])
                break
    if volume is None:
        print(f"  ERROR: Volume not found in {log_no_ions}")
        return False
    print(f"  Box volume: {volume:.2f} A^3")

    # 8. Calculate ions and run tleap with ions
    num_ions = calculate_num_ions(volume, 0.15, concentration)
    print(f"  Adding {num_ions} Na+ and {num_ions} Cl- ions")

    tleap_with_ions = write_tleap_input_with_ions(
        ff, water, water_box, size_box, edited_pdb, output_prefix,
        num_ions, ssbond_lines
    )
    log_with_ions = f"{output_prefix}_with_ions.log"
    if not run_tleap(tleap_with_ions, log_with_ions):
        return False

    print(f"  Successfully processed {pdb_file}")
    return True


# ======================== MAIN ========================
if __name__ == "__main__":
    args = parse_topology_arguments()

    ff_settings = FF_CONFIG[args.ff]
    ff        = ff_settings["leaprc"]
    water     = ff_settings["water"]
    water_box = ff_settings["water_box"]

    size_box        = args.size_box
    concentration   = args.concentration
    ph              = args.ph
    ssbond_complex  = args.ssbond_complex
    ion_aminoacid_list = args.ion_aminoacid_list

    input_folder = "./"
    work_dir     = "./"
    os.makedirs(work_dir, exist_ok=True)

    for pdb_file in os.listdir(input_folder):
        if pdb_file.endswith(".pdb"):
            pdb_path = os.path.join(input_folder, pdb_file)
            output_prefix = os.path.join(
                work_dir, os.path.splitext(pdb_file)[0]
            )
            work_dir_pdb = os.path.join(
                output_prefix, os.path.splitext(pdb_file)[0]
            )
            process_pdb(
                pdb_path, work_dir_pdb, ff, water, water_box, size_box,
                concentration, ph, ssbond_complex, ion_aminoacid_list,
                work_dir
            )
