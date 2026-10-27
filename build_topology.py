#!/usr/bin/env python3
# ============================================================
# build_topology.py
#
# AMBER topology generation for peptide + protein complexes,
# with SSBOND control and ionization-state overrides.
#
# Pipeline (per PDB file found in --input_dir):
#   1. parse SSBOND records from the original PDB
#   2. pdb2pqr  -> pH-dependent protonation states (PROPKA)
#   3. residue-name edits (inter-chain CYX->CYS, user overrides)
#   4. PyMOL    -> structural cleanup
#   5. BioPandas-> strip waters, OXT and hydrogens (TER kept)
#   6. tleap    -> solvate, neutralise, add salt, write prmtop/inpcrd
#
# Usage:
#   python build_topology.py --ff ff14SB --ph 7.0 --ssbond_complex no
#   python build_topology.py --ff ff19SB --ph 6.5 --ssbond_complex yes \
#                            --ion_aminoacid_list ionization_aminoacid.txt
# ============================================================

import os
import shutil
import subprocess
import argparse
import warnings
warnings.filterwarnings('ignore')

from pymol import cmd
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
        description="AMBER topology generation for peptide + protein "
                    "complexes, with SSBOND and ionization control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ff", choices=list(FF_CONFIG.keys()), default="ff14SB",
        help="Protein force field (ff14SB -> TIP3P, ff19SB -> OPC)"
    )
    parser.add_argument(
        "--size_box", type=float, default=10.0,
        help="Solvation buffer between solute and box edge (Angstrom)"
    )
    parser.add_argument(
        "--concentration", type=float, default=0.15,
        help="Salt concentration added on top of neutralisation (M)"
    )
    parser.add_argument(
        "--ph", type=float, default=7.0,
        help="System pH, used by PROPKA to assign titration states"
    )
    parser.add_argument(
        "--titration_method", choices=["propka", "none"], default="propka",
        help=(
            "How pdb2pqr assigns protonation states. 'propka' honours --ph; "
            "'none' uses the force-field default states and IGNORES --ph."
        )
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
            "Path to a file with user-preferred ionization states. "
            "Residue numbers refer to the ORIGINAL PDB numbering. "
            "Format: RESNAME RESID [CHAIN], one per line. "
            "Overrides the pdb2pqr/PROPKA assignment."
        )
    )
    parser.add_argument(
        "--input_dir", type=str, default="./",
        help="Directory scanned for input .pdb files"
    )
    parser.add_argument(
        "--work_dir", type=str, default="./",
        help="Directory where the per-complex output folders are created"
    )
    parser.add_argument(
        "--pdb", type=str, nargs="+", default=None,
        help="Process only these PDB files instead of every .pdb in "
             "--input_dir (names or paths)"
    )
    parser.add_argument(
        "--pdb2pqr_exe", type=str, default=None,
        help="pdb2pqr executable (auto-detected: pdb2pqr30, then pdb2pqr)"
    )
    parser.add_argument(
        "--tleap_exe", type=str, default=None,
        help="tleap executable (auto-detected)"
    )

    args = parser.parse_args()

    if args.concentration < 0:
        raise ValueError("Salt concentration must be non-negative")
    if not (0.0 <= args.ph <= 14.0):
        raise ValueError("pH must be between 0 and 14")
    if args.size_box <= 0:
        raise ValueError("Solvation box buffer must be positive")

    # Convert ssbond_complex from string to bool
    args.ssbond_complex = (args.ssbond_complex.lower() == "yes")

    return args


def resolve_executable(candidates, override=None, what="executable"):
    """Return the first candidate found on PATH, or the user override."""
    if override:
        if shutil.which(override) is None and not os.path.isfile(override):
            raise FileNotFoundError(f"{what} not found: {override}")
        return override
    for name in candidates:
        if shutil.which(name) is not None:
            return name
    raise FileNotFoundError(
        f"Could not find {what} on PATH (tried: {', '.join(candidates)}). "
        f"Activate the environment that provides it, or pass the path "
        f"explicitly."
    )


# ======================== PARSING FUNCTIONS ========================
def parse_ssbond_records(pdb_file):
    """Parse SSBOND records from PDB file header.

    Uses the fixed PDB columns defined by the format specification and
    falls back to token splitting for non-conforming files.

    Returns
    -------
    list of dict : {'chain1', 'resid1', 'chain2', 'resid2', 'is_interchain'}
    """
    ssbonds = []
    with open(pdb_file, 'r') as f:
        for line in f:
            # Stop once we reach coordinate records
            if line.startswith(('ATOM', 'HETATM', 'MODEL')):
                break
            if not line.startswith('SSBOND'):
                continue

            chain1 = resid1 = chain2 = resid2 = None
            # Preferred: fixed columns (chain 16/30, resSeq 18-21/32-35)
            if len(line) >= 35:
                try:
                    chain1 = line[15].strip()
                    resid1 = int(line[17:21])
                    chain2 = line[29].strip()
                    resid2 = int(line[31:35])
                except ValueError:
                    chain1 = None
            # Fallback: locate the two CYS tokens
            if chain1 is None:
                parts = line.split()
                cys_idx = [i for i, p in enumerate(parts) if p == 'CYS']
                if len(cys_idx) < 2:
                    continue
                try:
                    chain1 = parts[cys_idx[0] + 1]
                    resid1 = int(parts[cys_idx[0] + 2])
                    chain2 = parts[cys_idx[1] + 1]
                    resid2 = int(parts[cys_idx[1] + 2])
                except (IndexError, ValueError):
                    print(f"    WARNING: could not parse SSBOND record, "
                          f"skipping: {line.rstrip()}")
                    continue

            ssbonds.append({
                'chain1': chain1, 'resid1': resid1,
                'chain2': chain2, 'resid2': resid2,
                'is_interchain': chain1 != chain2,
            })
    return ssbonds


def parse_ionization_preferences(ion_file):
    """Parse user ionization preference file.

    Residue numbers refer to the ORIGINAL PDB numbering.
    The code translates them to the processed PDB numbering
    (which may differ if pdb2pqr renumbers residues).

    Format (one per line, tab- or space-separated):
        RESNAME RESID           -> applies to all chains
        RESNAME RESID CHAIN     -> chain-specific
    Lines starting with '#' are comments.

    Returns
    -------
    list of dict : {'resname', 'resid', 'chain' (optional)}
    """
    preferences = []
    if ion_file is None:
        return preferences
    if not os.path.exists(ion_file):
        print(f"  WARNING: ionization list '{ion_file}' not found "
              f"-- no overrides will be applied")
        return preferences
    with open(ion_file, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"  WARNING: {ion_file}:{lineno}: expected "
                      f"'RESNAME RESID [CHAIN]', got '{line}' -- skipping")
                continue
            try:
                resid = int(parts[1])
            except ValueError:
                print(f"  WARNING: {ion_file}:{lineno}: residue number "
                      f"'{parts[1]}' is not an integer -- skipping")
                continue
            pref = {'resname': parts[0].upper(), 'resid': resid}
            if len(parts) > 2:
                pref['chain'] = parts[2]
            preferences.append(pref)
    return preferences


def _ordered_residues(pdb_file):
    """Ordered, unique (chain_id, residue_number, residue_name) of ATOM records."""
    ppdb = PandasPdb().read_pdb(pdb_file)
    atoms = ppdb.df['ATOM']
    cols = ['chain_id', 'residue_number', 'residue_name']
    residues = atoms[cols].drop_duplicates()
    return [(row['chain_id'], int(row['residue_number']), row['residue_name'])
            for _, row in residues.iterrows()]


def build_pdb_residue_mapping(original_pdb, processed_pdb):
    """Build a mapping original PDB (chain, resid) -> processed PDB (chain, resid).

    Matches residues by their sequential position in the file, which handles
    any renumbering applied by pdb2pqr. Residue names are compared as a
    sanity check, allowing for protonation-state renaming.

    Returns
    -------
    dict : {(orig_chain, orig_resid): (proc_chain, proc_resid)}
    """
    orig_residues = _ordered_residues(original_pdb)
    proc_residues = _ordered_residues(processed_pdb)

    if len(orig_residues) != len(proc_residues):
        print(f"    WARNING: residue count mismatch -- original "
              f"{len(orig_residues)}, processed {len(proc_residues)}. "
              f"Residues dropped by pdb2pqr (non-standard residues, "
              f"alternate locations) shift the mapping; check the "
              f"overrides reported below.")

    mapping = {}
    n_name_mismatch = 0
    for (o_chain, o_resid, o_name), (p_chain, p_resid, p_name) in zip(
            orig_residues, proc_residues):
        mapping[(o_chain, o_resid)] = (p_chain, p_resid)
        family = IONIZATION_FAMILIES.get(p_name, {p_name})
        if o_name not in family and o_name != p_name:
            n_name_mismatch += 1
            if n_name_mismatch <= 5:
                print(f"    WARNING: mapping ({o_chain}, {o_resid}) {o_name} "
                      f"-> ({p_chain}, {p_resid}) {p_name}: residue names "
                      f"are unrelated, the mapping may be misaligned")
    if n_name_mismatch > 5:
        print(f"    WARNING: {n_name_mismatch} residue-name mismatches in "
              f"total while mapping original -> processed numbering")

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

    # Save modified PDB (overwrite), keeping TER records
    write_pdb_with_ter(ppdb, pdb_file)
    print(f"    Residue modifications: {n_reverted} CYX->CYS reverted, "
          f"{n_overrides} ionization overrides applied")


# ======================== PDB WRITING ========================
def write_pdb_with_ter(ppdb, path):
    """Write ATOM/HETATM records and re-insert the TER records.

    BioPandas keeps TER lines in its 'OTHERS' dataframe, so writing only
    ATOM/HETATM silently drops every chain terminator. tleap then bonds the
    last residue of one chain to the first residue of the next, fusing the
    peptide and the receptor into a single covalent chain, so the TER
    records have to be restored.

    A TER is inserted whenever the chain id changes and at the ATOM ->
    HETATM transition. Consecutive HETATM residues are left untouched, so
    multi-residue ligands and glycans stay connected.
    """
    ppdb.to_pdb(path=path, records=['ATOM', 'HETATM'],
                gz=False, append_newline=True)

    with open(path, 'r') as handle:
        lines = [l.rstrip('\n') for l in handle if l.strip()]

    def ter_for(atom_line):
        """Build a spec-compliant TER record from the preceding atom line."""
        try:
            serial = int(atom_line[6:11]) + 1
        except ValueError:
            serial = 0
        return f"TER   {serial:>5}      {atom_line[17:27]}"

    out = []
    prev_atom = None    # last ATOM/HETATM line seen
    prev_key = None     # (record_name, chain_id)
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            record = 'ATOM' if line.startswith('ATOM') else 'HETATM'
            key = (record, line[21])
            if prev_key is not None and key != prev_key:
                out.append(ter_for(prev_atom))
            prev_key, prev_atom = key, line
        out.append(line)
    if prev_atom is not None:
        out.append(ter_for(prev_atom))
    out.append('END')

    with open(path, 'w') as handle:
        handle.write('\n'.join(out) + '\n')


def warn_on_numbering_gaps(pdb_file):
    """Warn about residue-number gaps inside a chain.

    tleap bonds consecutive residues of a chain regardless of distance, so
    a missing loop in the input structure becomes an unphysical long bond.
    """
    ppdb = PandasPdb().read_pdb(pdb_file)
    atoms = ppdb.df['ATOM']
    residues = atoms[['chain_id', 'residue_number']].drop_duplicates()
    prev_chain, prev_resid = None, None
    for _, row in residues.iterrows():
        chain, resid = row['chain_id'], int(row['residue_number'])
        if chain == prev_chain and resid - prev_resid > 1:
            print(f"    WARNING: chain {chain} jumps from residue "
                  f"{prev_resid} to {resid} (missing residues). tleap will "
                  f"bond across the gap -- model the loop or split the "
                  f"chain if that is not what you want.")
        prev_chain, prev_resid = chain, resid


# ======================== PDB PREPROCESSING ========================
def preprocess_pdb(pdb_file, work_dir, ph, titration_method, ssbonds,
                   ssbond_complex, ionization_prefs, pdb2pqr_exe):
    """Full PDB preprocessing pipeline:
    1. pdb2pqr  ->  protonation-state assignment
    2. Build residue mapping (original -> pdb2pqr numbering)
    3. Residue modifications  ->  CYX->CYS + ionization overrides
    4. PyMOL    ->  structural cleanup
    5. BioPandas ->  remove water, OXT, hydrogens (TER preserved)

    Returns
    -------
    (str, dict) : path of the cleaned PDB, original -> processed mapping
    """
    name = os.path.splitext(os.path.basename(pdb_file))[0]
    pdb_output_dir = os.path.join(work_dir, name)
    os.makedirs(pdb_output_dir, exist_ok=True)

    pdb2pqr_output = os.path.join(pdb_output_dir, f"{name}_pka")

    # Step 1: pdb2pqr.
    # NOTE - '--with-ph' is silently ignored unless a titration-state method
    # is requested, so PROPKA has to be asked for explicitly.
    pdb2pqr_cmd = [
        pdb2pqr_exe, pdb_file,
        "--ff", "AMBER", "--ffout", "AMBER",
        "--pdb-output", f"{pdb2pqr_output}.pdb",
    ]
    if titration_method == "propka":
        pdb2pqr_cmd += ["--titration-state-method", "propka",
                        "--with-ph", str(ph)]
        print(f"  Running pdb2pqr with PROPKA (pH={ph}) ...")
    else:
        print(f"  Running pdb2pqr with force-field default states "
              f"(--ph is ignored) ...")
    pdb2pqr_cmd.append(f"{pdb2pqr_output}.pqr")
    subprocess.run(pdb2pqr_cmd, check=True)

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

    # Step 4: PyMOL cleanup.
    # retain_order has to be set before loading for it to take effect.
    print(f"  PyMOL cleanup ...")
    obj_name = f"sys_{name}"
    cmd.set("retain_order", "on")
    cmd.load(f"{pdb2pqr_output}.pdb", obj_name)
    pymol_output = os.path.join(pdb_output_dir, f"{name}_pymol.pdb")
    cmd.save(pymol_output, obj_name, format="pdb")
    cmd.delete(obj_name)

    # Step 5: BioPandas cleanup (remove OXT, H, water)
    print(f"  BioPandas cleanup ...")
    edited_pdb_path = os.path.join(pdb_output_dir, f"{name}_ed.pdb")
    ppdb = PandasPdb().read_pdb(pymol_output)
    atoms = ppdb.df['ATOM']
    atoms = atoms[atoms['atom_name'] != 'OXT']
    # Prefer the element column; fall back to atom names when it is blank,
    # otherwise hydrogens would survive into tleap.
    element = atoms['element_symbol'].astype(str).str.strip().str.upper()
    atom_name = atoms['atom_name'].astype(str).str.strip()
    is_hydrogen = (element == 'H') | (
        (element == '') & atom_name.str.match(r'^[0-9]?H')
    )
    ppdb.df['ATOM'] = atoms[~is_hydrogen]
    ppdb.df['HETATM'] = ppdb.df['HETATM'][
        ~ppdb.df['HETATM']['residue_name'].isin(['HOH', 'WAT', 'SOL'])
    ]
    write_pdb_with_ter(ppdb, edited_pdb_path)
    warn_on_numbering_gaps(edited_pdb_path)

    n_chains = sum(1 for line in open(edited_pdb_path)
                   if line.startswith('TER'))
    print(f"  Cleaned PDB has {n_chains} chain(s)/molecule(s) "
          f"(TER records written)")

    return edited_pdb_path, orig_to_proc_mapping


# ======================== TLEAP FUNCTIONS ========================
def get_ssbond_tleap_lines(ssbonds, ssbond_complex, residue_mapping,
                           orig_to_proc_mapping):
    """Generate tleap 'bond' commands for disulfide bonds.

    - Intra-chain SSBONDs are ALWAYS included.
    - Inter-chain SSBONDs are included ONLY if ssbond_complex=True.
    - SSBOND records use the ORIGINAL numbering, so they are translated
      through orig_to_proc_mapping before being looked up in the final
      PDB's sequential tleap index.
    """
    bond_lines = []
    for ss in ssbonds:
        if ss['is_interchain'] and not ssbond_complex:
            continue
        keys = []
        for chain, resid in [(ss['chain1'], ss['resid1']),
                             (ss['chain2'], ss['resid2'])]:
            keys.append(orig_to_proc_mapping.get((chain, resid),
                                                 (chain, resid)))
        idx1 = residue_mapping.get(keys[0])
        idx2 = residue_mapping.get(keys[1])
        if idx1 is not None and idx2 is not None:
            bond_lines.append(f"bond SYS.{idx1}.SG SYS.{idx2}.SG")
        else:
            missing = [k for k, v in [(keys[0], idx1), (keys[1], idx2)]
                       if v is None]
            print(f"  WARNING: SSBOND residue(s) {missing} not found in "
                  f"final PDB -- skipping bond")
    return bond_lines


def _write_tleap_header(f, ff, water, pdb_file, ssbond_lines):
    f.write(f"source {ff}\n")
    f.write("source leaprc.DNA.OL15\n")
    f.write("source leaprc.RNA.OL3\n")
    f.write("source leaprc.GLYCAM_06j-1\n")
    f.write("source leaprc.gaff2\n")
    f.write(f"source {water}\n")
    f.write(f"SYS = loadpdb {pdb_file}\n")
    for line in ssbond_lines:
        f.write(f"{line}\n")


def write_tleap_input_no_ions(ff, water, water_box, size_box,
                              pdb_file, output_prefix, ssbond_lines):
    """Write tleap input without ions (used for volume estimation)."""
    tleap_file = f"{output_prefix}_no_ions_tleap.in"
    with open(tleap_file, "w") as f:
        _write_tleap_header(f, ff, water, pdb_file, ssbond_lines)
        f.write("alignaxes SYS\n")
        f.write(f"solvateOct SYS {water_box} {size_box}\n")
        f.write(f"saveamberparm SYS {output_prefix}_no_ions.prmtop "
                f"{output_prefix}_no_ions.inpcrd\n")
        f.write(f"savepdb SYS {output_prefix}_no_ions.pdb\n")
        f.write("quit\n")
    return tleap_file


def write_tleap_input_with_ions(ff, water, water_box, size_box,
                                pdb_file, output_prefix, num_ions,
                                ssbond_lines):
    """Write tleap input with neutralization + ion concentration."""
    tleap_file = f"{output_prefix}_with_ions_tleap.in"
    with open(tleap_file, "w") as f:
        _write_tleap_header(f, ff, water, pdb_file, ssbond_lines)
        f.write("alignaxes SYS\n")
        f.write("check SYS\n")
        f.write("charge SYS\n")
        f.write("addions SYS Na+ 0\n")
        f.write("addions SYS Cl- 0\n")
        f.write("check SYS\n")
        f.write("charge SYS\n")
        f.write(f"solvateOct SYS {water_box} {size_box}\n")
        if num_ions > 0:
            f.write(f"addIonsRand SYS Na+ {num_ions} Cl- {num_ions}\n")
        f.write(f"saveamberparm SYS {output_prefix}_amber.prmtop "
                f"{output_prefix}_amber.inpcrd\n")
        f.write(f"savepdb SYS {output_prefix}_amber.pdb\n")
        f.write("quit\n")
    return tleap_file


def calculate_num_ions(volume, base_concentration, target_concentration):
    """Calculate number of ions from box volume and target salt concentration."""
    vol_liters = volume * 1e-27
    atom_liters = 9.03e22
    return int(vol_liters * (target_concentration / base_concentration)
               * atom_liters)


def run_tleap(tleap_exe, tleap_input, tleap_log):
    """Execute tleap with the given input file.

    tleap frequently exits with status 0 even after a fatal error, so the
    log is inspected as well.
    """
    try:
        with open(tleap_log, 'w') as log:
            subprocess.run([tleap_exe, "-f", tleap_input],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR running tleap: {e} (see {tleap_log})")
        return False

    with open(tleap_log) as log:
        fatal = [l.rstrip() for l in log if 'FATAL' in l or 'Fatal' in l]
    if fatal:
        print(f"  ERROR: tleap reported a fatal error (see {tleap_log}):")
        for line in fatal[:5]:
            print(f"    {line}")
        return False
    return True


# ======================== MAIN PIPELINE ========================
def process_pdb(pdb_file, output_prefix, ff, water, water_box, size_box,
                concentration, ph, titration_method, ssbond_complex,
                ion_aminoacid_list, work_dir, pdb2pqr_exe, tleap_exe):
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
    edited_pdb, orig_to_proc_mapping = preprocess_pdb(
        pdb_file, work_dir, ph, titration_method, ssbonds, ssbond_complex,
        ionization_prefs, pdb2pqr_exe
    )
    print(f"  Final cleaned PDB: {edited_pdb}")

    # 4. Build tleap residue mapping from the final cleaned PDB
    residue_mapping = build_tleap_residue_mapping(edited_pdb)
    print(f"  Residue mapping built: {len(residue_mapping)} residues")

    # 5. Generate SSBOND tleap bond lines
    ssbond_lines = get_ssbond_tleap_lines(
        ssbonds, ssbond_complex, residue_mapping, orig_to_proc_mapping
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
    if not run_tleap(tleap_exe, tleap_no_ions, log_no_ions):
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
    print(f"  Adding {num_ions} Na+ and {num_ions} Cl- ions "
          f"on top of neutralisation")

    tleap_with_ions = write_tleap_input_with_ions(
        ff, water, water_box, size_box, edited_pdb, output_prefix,
        num_ions, ssbond_lines
    )
    log_with_ions = f"{output_prefix}_with_ions.log"
    if not run_tleap(tleap_exe, tleap_with_ions, log_with_ions):
        return False

    # 9. Make sure tleap really wrote the files the next stage needs
    expected = [f"{output_prefix}_amber.prmtop",
                f"{output_prefix}_amber.inpcrd"]
    missing = [p for p in expected
               if not (os.path.isfile(p) and os.path.getsize(p) > 0)]
    if missing:
        print(f"  ERROR: tleap did not write {missing} "
              f"(see {log_with_ions})")
        return False

    print(f"  Successfully processed {pdb_file}")
    print(f"    topology   : {expected[0]}")
    print(f"    coordinates: {expected[1]}")
    return True


# ======================== MAIN ========================
def main():
    args = parse_topology_arguments()

    pdb2pqr_exe = resolve_executable(["pdb2pqr30", "pdb2pqr"],
                                     args.pdb2pqr_exe, "pdb2pqr")
    tleap_exe = resolve_executable(["tleap"], args.tleap_exe, "tleap")

    ff_settings = FF_CONFIG[args.ff]
    ff        = ff_settings["leaprc"]
    water     = ff_settings["water"]
    water_box = ff_settings["water_box"]

    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)

    if args.pdb:
        pdb_paths = []
        for entry in args.pdb:
            path = entry if os.path.isfile(entry) else os.path.join(
                args.input_dir, entry)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"PDB file not found: {entry}")
            pdb_paths.append(path)
    else:
        pdb_paths = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir) if f.endswith(".pdb")
        )

    if not pdb_paths:
        print(f"No .pdb files found in '{args.input_dir}'. Nothing to do.")
        return

    print(f"Force field: {args.ff} ({ff} + {water})")
    print(f"pdb2pqr: {pdb2pqr_exe}   tleap: {tleap_exe}")
    print(f"Found {len(pdb_paths)} PDB file(s) to process")

    succeeded, failed = [], []
    for pdb_path in pdb_paths:
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        # Everything for one complex lives in '<work_dir>/<name>/'
        output_prefix = os.path.join(work_dir, name, name)
        try:
            ok = process_pdb(
                pdb_path, output_prefix, ff, water, water_box, args.size_box,
                args.concentration, args.ph, args.titration_method,
                args.ssbond_complex, args.ion_aminoacid_list, work_dir,
                pdb2pqr_exe, tleap_exe
            )
        except Exception as exc:
            # Keep going through the rest of the batch
            print(f"  ERROR while processing {name}: {exc}")
            ok = False
        (succeeded if ok else failed).append(name)

    print(f"\n{'='*60}")
    print(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print(f"  failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
