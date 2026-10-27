#!/usr/bin/env python3
# ============================================================
# run_bpmd.py
#
# Open Binding Pose Metadynamics (OpenBPMD) for peptide ligands:
# evaluating peptide pose stability on a receptor using metadynamics.
#
# OpenBPMD is an open source implementation of binding pose metadynamics
# (BPMD), based on the work by Clark et al, 2016
# (https://doi.org/10.1021/acs.jctc.6b00201).
#
# Script developed by @pablitoarantes
#
# The peptide RMSD is biased with well-tempered metadynamics. A peptide
# that stays in its starting pose gives a low PoseScore and keeps its
# receptor contacts; a weakly bound peptide is pushed out of the pose by
# the bias.
#
# Usage:
#   python run_bpmd.py --work_dir ./ --nreps 10
#   python run_bpmd.py --work_dir ./ --complex mycomplex --platform CUDA
# ============================================================

from openmm.app.metadynamics import Metadynamics, BiasVariable
import openmm as mm
from openmm import (LangevinIntegrator, HarmonicBondForce, NonbondedForce,
                    CustomCentroidBondForce, RMSDForce, Platform)
from openmm.app import (AmberInpcrdFile, AmberPrmtopFile, PDBFile,
                        Simulation, DCDReporter, StateDataReporter,
                        PME, HBonds)
from openmm.unit import (kelvin, picosecond, picoseconds, nanometer,
                         nanometers, kilocalories_per_mole,
                         kilojoules_per_mole, angstrom, amu)
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts
import mdtraj as md
import pandas as pd
import glob
import os
import re
import sys
from sys import stdout

# ---------------------------------------------------------------------
# Fixed protocol constants. These define the BPMD protocol itself and are
# deliberately not exposed on the command line.
# ---------------------------------------------------------------------
TIMESTEP_PS = 0.002          # 2 fs integration timestep
FRICTION = 1.0               # Langevin friction coefficient, 1/ps
RESTRAINT_K = 5              # equilibration restraint, kcal/mol/A^2
BIAS_FREQUENCY = 250         # deposit a hill every 250 steps (0.5 ps)
BIAS_FACTOR = 4.0            # well-tempered metadynamics bias factor
HILL_WIDTH = 0.002           # metadynamics sigma, nm
GRID_MIN, GRID_MAX = 0.0, 1.0   # RMSD CV range, nm
N_FRAMES = 100               # trajectory frames written per replica
COLVAR_CHUNK = 500           # steps between CV records (1 ps)
CONTACT_RADIUS = 3.5         # native-contact cutoff, Angstrom
CONTACT_SHELL = 5            # receptor shell around the peptide, Angstrom
COMP_CONTACT_WEIGHT = 5      # CompScore = PoseScore - 5 * ContactScore
SCORE_WINDOW_NS = 2.0        # final scores average the last 2 ns
CONTACT_REF_NS = 1.0         # contacts normalised by the first 1 ns


# ======================== ARGUMENT PARSING ========================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Open Binding Pose Metadynamics for peptide + protein "
                    "complexes (OpenBPMD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    io = parser.add_argument_group("input / output")
    io.add_argument(
        "--work_dir", type=str, default="./",
        help="Directory holding one sub-directory per complex, as produced "
             "by build_topology.py"
    )
    io.add_argument(
        "--complex", type=str, nargs="+", default=None,
        help="Names of the complexes to run (the sub-directory names). "
             "Default: every sub-directory that contains a topology"
    )
    io.add_argument(
        "--output", type=str, default="peptide_pose",
        help="Name of the output directory created inside each complex "
             "directory"
    )
    io.add_argument(
        "--prmtop_suffix", type=str, default="_amber.prmtop",
        help="Suffix of the AMBER topology file, appended to the "
             "complex name"
    )
    io.add_argument(
        "--inpcrd_suffix", type=str, default="_amber.inpcrd",
        help="Suffix of the AMBER coordinate file"
    )
    io.add_argument(
        "--pdb_suffix", type=str, default="_ed.pdb",
        help="Suffix of the dry solute PDB used to detect the peptide and "
             "receptor residue ranges"
    )

    sel = parser.add_argument_group("system definition")
    sel.add_argument(
        "--lig_chain", type=int, default=1,
        help="1-based index of the chain that is the PEPTIDE (the biased "
             "ligand), counted in the order the chains appear in the PDB"
    )
    sel.add_argument(
        "--prot_chain", type=int, default=2,
        help="1-based index of the chain that is the RECEPTOR"
    )

    sim = parser.add_argument_group("simulation protocol")
    sim.add_argument(
        "--nreps", type=int, default=10,
        help="Number of repeat OpenBPMD simulations to run in series"
    )
    sim.add_argument(
        "--hill_height", type=float, default=0.5,
        help="Height of the metadynamics hill, in kcal/mol"
    )
    sim.add_argument(
        "--sim_time", type=float, default=10.0,
        help="Length of each production replica, in ns"
    )
    sim.add_argument(
        "--equil_time", type=float, default=1000.0,
        help="Length of the restrained NVT equilibration, in ps"
    )
    sim.add_argument(
        "--temperature", type=float, default=300.0,
        help="Simulation temperature, in K"
    )

    hw = parser.add_argument_group("hardware")
    hw.add_argument(
        "--platform", type=str, default=None,
        choices=["CUDA", "OpenCL", "HIP", "CPU", "Reference"],
        help="OpenMM platform. Default: let OpenMM pick the fastest one"
    )
    hw.add_argument(
        "--precision", type=str, default="mixed",
        choices=["single", "mixed", "double"],
        help="Precision for the CUDA / OpenCL / HIP platforms"
    )
    hw.add_argument(
        "--device_index", type=str, default=None,
        help="GPU device index for CUDA / OpenCL / HIP, e.g. '0' or '0,1'"
    )

    args = parser.parse_args()

    if args.nreps < 1:
        raise ValueError("--nreps must be at least 1")
    if args.sim_time <= 0:
        raise ValueError("--sim_time must be positive")
    if args.equil_time <= 0:
        raise ValueError("--equil_time must be positive")
    if args.hill_height <= 0:
        raise ValueError("--hill_height must be positive")
    if args.lig_chain == args.prot_chain:
        raise ValueError("--lig_chain and --prot_chain must differ")
    if args.lig_chain < 1 or args.prot_chain < 1:
        raise ValueError("chain indices are 1-based, so they must be >= 1")
    if args.sim_time < SCORE_WINDOW_NS:
        print(f"WARNING: --sim_time {args.sim_time} ns is shorter than the "
              f"{SCORE_WINDOW_NS} ns scoring window; the final scores will "
              f"average the whole trajectory.")

    return args


def get_platform(args):
    """Return (platform, properties) for the Simulation constructor."""
    if args.platform is None:
        return None, None
    platform = Platform.getPlatformByName(args.platform)
    properties = {}
    if args.platform in ("CUDA", "OpenCL", "HIP"):
        prefix = {"CUDA": "Cuda", "OpenCL": "OpenCL", "HIP": "Hip"}[
            args.platform]
        properties[f"{prefix}Precision"] = args.precision
        if args.device_index is not None:
            properties[f"{prefix}DeviceIndex"] = args.device_index
    return platform, properties


def make_simulation(topology, system, integrator, platform, properties):
    """Create a Simulation, honouring an explicit platform if given."""
    if platform is None:
        return Simulation(topology, system, integrator)
    if properties:
        return Simulation(topology, system, integrator, platform, properties)
    return Simulation(topology, system, integrator, platform)


# ======================== STRUCTURE INSPECTION ========================
def chain_ranges_from_pdb(pdb_file):
    """Find the residue range of every chain / molecule in a PDB file.

    Ranges are expressed as SEQUENTIAL 1-based residue indices counted
    across the whole file, because that is how both tleap and OpenMM
    number the residues of the resulting topology, and therefore what
    MDAnalysis sees as 'resid' in the simulation output.

    Chains are split on TER records and, as a fallback for files written
    without them, on a change of chain id.

    Returns
    -------
    list of (start, end) tuples, one per chain / molecule.
    """
    ranges = []
    count = 0            # sequential residue counter
    start = 1            # first sequential index of the current chain
    prev_res = None      # (chain_id, resSeq + insertion code)
    prev_chain = None

    def close():
        nonlocal start, prev_res, prev_chain
        if count >= start:
            ranges.append((start, count))
            start = count + 1
        prev_res, prev_chain = None, None

    with open(pdb_file) as handle:
        for line in handle:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                chain = line[21]
                res_key = (chain, line[22:27])
                # a change of chain id also ends a molecule
                if prev_chain is not None and chain != prev_chain:
                    close()
                prev_chain = chain
                if res_key != prev_res:
                    count += 1
                    prev_res = res_key
            elif record == "TER":
                close()
            elif record in ("END", "ENDMDL"):
                break
    close()

    return ranges


def first_residue_number(pdb_file):
    """Residue number of the first ATOM/HETATM record of a PDB file."""
    with open(pdb_file) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                return int(line[22:26])
    raise ValueError(f"No ATOM records found in {pdb_file}")


def check_selection(universe, selection, what):
    """Make sure a selection matches atoms, with a helpful error if not."""
    n = len(universe.select_atoms(selection))
    if n == 0:
        raise ValueError(
            f"The {what} selection '{selection}' matches no atoms. "
            f"Check --lig_chain / --prot_chain against the chains detected "
            f"in the input PDB."
        )
    return n


# ======================== ANALYSIS ========================
def get_contact_score(structure_file, trajectory_file, lig_resname,
                      lig_resname_analysis, prot_resname, sim_time):
    """A function that gets the ContactScore from an OpenBPMD trajectory.

    The number of peptide-receptor contacts in every frame is normalised
    by the average over the first nanosecond, and capped at 1.

    Parameters
    ----------
    structure_file : str
        The name of the centred equilibrated system PDB file that
        was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Selection string for the peptide that was biased.
    lig_resname_analysis : str
        Selection string for the peptide, used to define the receptor
        shell around it.
    prot_resname : str
        Selection string for the receptor.
    sim_time : float
        Length of the trajectory, in ns, used to locate the first
        nanosecond of frames.

    Returns
    -------
    contact_scores : np.array
        ContactScore for every frame of the trajectory.
    """
    u = mda.Universe(structure_file, trajectory_file)

    # NOTE - 'not name H*' and not 'not name *H*': the latter would also
    # drop heavy atoms whose name contains an H, such as Arg NH1/NH2 and
    # Tyr OH, which are exactly the atoms that make the contacts.
    sel_donor = f"{lig_resname} and not name H*"

    sel_acceptor = f"{prot_resname} and not name H* and \
                     around {CONTACT_SHELL} {lig_resname_analysis}"

    check_selection(u, sel_donor, "peptide contact")
    check_selection(u, sel_acceptor, "receptor contact")

    # reference groups (first frame of the trajectory, but you could also use
    # a separate PDB, eg crystal structure)
    a_donors = u.select_atoms(sel_donor)
    a_acceptors = u.select_atoms(sel_acceptor)

    cont_analysis = contacts.Contacts(u, select=(sel_donor, sel_acceptor),
                                      refgroup=(a_donors, a_acceptors),
                                      radius=CONTACT_RADIUS)

    cont_analysis.run()
    # Average the contacts over the first nanosecond and use that as the
    # reference the rest of the trajectory is normalised against.
    n_frames = len(cont_analysis.timeseries)
    frame_idx_first_ns = max(2, int(n_frames * CONTACT_REF_NS / sim_time))
    frame_idx_first_ns = min(frame_idx_first_ns, n_frames)
    first_ns_mean = np.mean(cont_analysis.timeseries[1:frame_idx_first_ns, 1])
    if first_ns_mean == 0:
        normed_contacts = cont_analysis.timeseries[1:, 1]
    else:
        normed_contacts = cont_analysis.timeseries[1:, 1]/first_ns_mean
    contact_scores = np.where(normed_contacts > 1, 1, normed_contacts)

    return contact_scores


def get_pose_score(structure_file, trajectory_file, lig_resname):
    """A function that gets the PoseScore (peptide RMSD) from an OpenBPMD
    trajectory.

    Parameters
    ----------
    structure_file : str
        The name of the centred equilibrated system
        PDB file that was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Selection string for the peptide that was biased.

    Returns
    -------
    pose_scores : np.array
        PoseScore for every frame of the trajectory.
    """
    # Load a MDA universe with the trajectory
    u = mda.Universe(structure_file, trajectory_file)
    check_selection(u, f'{lig_resname} and not name H*', "peptide RMSD")
    # Align each frame using the backbone as reference
    # Calculate the RMSD of peptide heavy atoms
    r = rms.RMSD(u, select='backbone',
                 groupselections=[f'{lig_resname} and not name H*'],
                 ref_frame=0).run()
    # Get the PoseScores as np.array
    pose_scores = r.rmsd[1:, -1]

    return pose_scores


def collect_results(in_dir, out_dir, sim_time):
    """A function that collects the time-resolved BPM results,
    takes the scores from last 2 ns of the simulation, averages them
    and writes that average as the final score for a given pose.

    Writes a 'results.csv' file with the averages over all replicas and a
    'results_per_replica.csv' file with one row per replica, in the
    'out_dir' directory.

    Parameters
    ----------
    in_dir : str
        Directory with 'rep_*' directories.
    out_dir : str
        Directory where the 'results.csv' file will be written
    sim_time : float
        Length of each replica, in ns, used to locate the last 2 ns.
    """
    compList = []
    contactList = []
    poseList = []
    per_replica = []

    # find the repeats that have actually finished
    def replica_number(path):
        """Sort rep_2 before rep_10, tolerating unexpected directory names."""
        match = re.search(r'rep_(\d+)$', os.path.basename(os.path.dirname(path)))
        return int(match.group(1)) if match else 0

    result_files = sorted(
        glob.glob(os.path.join(in_dir, 'rep_*', 'bpm_results.csv')),
        key=replica_number
    )
    if not result_files:
        print("  WARNING: no 'bpm_results.csv' files found, "
              "not writing results.csv")
        return

    for f in result_files:
        rep_name = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        if df.empty:
            print(f"  WARNING: {f} is empty -- skipping")
            continue
        # Since we only want the last 2 ns, get the index of
        # the corresponding fraction of the data points
        n = len(df['CompScore'].values)
        window = min(n, max(1, round(n * SCORE_WINDOW_NS / sim_time)))
        comp = df['CompScore'].values[-window:]
        contact = df['ContactScore'].values[-window:]
        pose = df['PoseScore'].values[-window:]
        compList.append(comp)
        contactList.append(contact)
        poseList.append(pose)
        per_replica.append({
            'replica': rep_name,
            'CompScore': comp.mean(), 'PoseScore': pose.mean(),
            'ContactScore': contact.mean(),
        })

    if not compList:
        print("  WARNING: no usable replica results, "
              "not writing results.csv")
        return

    # Pool every scored frame of every replica. np.concatenate keeps this
    # working if the replicas differ in length.
    comp_all = np.concatenate(compList)
    pose_all = np.concatenate(poseList)
    contact_all = np.concatenate(contactList)

    # Format it the Pandas way
    d = {'CompScore': [np.mean(comp_all)], 'CompScoreSD': [np.std(comp_all)],
         'PoseScore': [np.mean(pose_all)], 'PoseScoreSD': [np.std(pose_all)],
         'ContactScore': [np.mean(contact_all)],
         'ContactScoreSD': [np.std(contact_all)],
         'nreps': [len(compList)]}

    results_df = pd.DataFrame(data=d)
    results_df = results_df.round(3)
    results_df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)

    pd.DataFrame(per_replica).round(3).to_csv(
        os.path.join(out_dir, 'results_per_replica.csv'), index=False)

    print(f"  Final scores over {len(compList)} replica(s), "
          f"last {SCORE_WINDOW_NS} ns:")
    print(results_df.to_string(index=False))


# ======================== SIMULATION ========================
def minimize(parm, input_positions, out_dir, min_file_name,
             platform=None, properties=None):
    """An energy minimization function, down to the default OpenMM energy
    tolerance of 10 kJ/mol.

    Parameters
    ----------
    parm : OpenMM parameter file object
        Used to create the OpenMM System object.
    input_positions : OpenMM Quantity
        3D coordinates of the system.
    out_dir : str
        Directory to write the outputs.
    min_file_name : str
        Name of the minimized PDB file to write.
    platform, properties :
        Optional OpenMM platform and platform properties.
    """
    system = parm.createSystem(nonbondedMethod=PME,
                               nonbondedCutoff=1*nanometers,
                               constraints=HBonds)

    # Set up the simulation parameters
    # Langevin integrator at 300 K w/ 1 ps^-1 friction coefficient
    # and a 2-fs timestep
    # NOTE - no dynamics performed, but required for setting up
    # the OpenMM system.
    integrator = LangevinIntegrator(300*kelvin, FRICTION/picosecond,
                                    TIMESTEP_PS*picoseconds)
    simulation = make_simulation(parm.topology, system, integrator,
                                 platform, properties)
    simulation.context.setPositions(input_positions)

    # Minimize the system - no predefined number of steps
    simulation.minimizeEnergy()

    # Write out the minimized system to use w/ MDAnalysis
    positions = simulation.context.getState(getPositions=True).getPositions()
    out_file = os.path.join(out_dir, min_file_name)
    with open(out_file, 'w') as handle:
        PDBFile.writeFile(simulation.topology, positions, handle)

    return None


def equilibrate(min_pdb, parm, out_dir, eq_file_name, equil_time,
                temperature, platform=None, properties=None):
    """A function that does an NVT equilibration with position restraints,
    with a 5 kcal/mol/A**2 harmonic constant on solute heavy atoms, using
    a 2 fs timestep.

    Parameters
    ----------
    min_pdb : str
        Name of the minimized PDB file.
    parm : OpenMM parameter file object
        Used to create the OpenMM System object.
    out_dir : str
        Directory to write the outputs to.
    eq_file_name : str
        Name of the equilibrated PDB file to write.
    equil_time : float
        Length of the equilibration, in ps.
    temperature : float
        Thermostat temperature, in K.
    platform, properties :
        Optional OpenMM platform and platform properties.
    """
    # Get the solute heavy atom indices to use
    # for defining position restraints during equilibration
    universe = mda.Universe(min_pdb,
                            format='XPDB', in_memory=True)
    solute_heavy_atom_idx = universe.select_atoms('not resname WAT and\
                                                   not resname SOL and\
                                                   not resname HOH and\
                                                   not resname CL and \
                                                   not resname NA and \
                                                   not resname K and \
                                                   not name H*').indices
    # Necessary conversion to int from numpy.int64,
    # b/c it breaks OpenMM C++ function
    solute_heavy_atom_idx = [int(idx) for idx in solute_heavy_atom_idx]
    if not solute_heavy_atom_idx:
        raise ValueError(f"No solute heavy atoms found in {min_pdb}")

    # Add the restraints.
    # We add a dummy atoms with no mass, which are therefore unaffected by
    # any kind of scaling done by barostat (if used). And the atoms are
    # harmonically restrained to the dummy atom. We have to redefine the
    # system, b/c we're adding new particles and this would clash with
    # modeller.topology.
    system = parm.createSystem(nonbondedMethod=PME,
                               nonbondedCutoff=1*nanometers,
                               constraints=HBonds)

    # Add the harmonic restraints on the positions
    # of specified atoms
    restraint = HarmonicBondForce()
    restraint.setUsesPeriodicBoundaryConditions(True)
    system.addForce(restraint)
    nonbonded = [force for force in system.getForces()
                 if isinstance(force, NonbondedForce)][0]
    dummyIndex = []
    input_positions = PDBFile(min_pdb).getPositions()
    # Keep an unmodified copy: the dummy positions are appended to
    # 'input_positions' as we go, and reading them back from the same
    # object would be fragile.
    positions = list(input_positions)
    # Go through the indices of all atoms that will be restrained
    for i in solute_heavy_atom_idx:
        j = system.addParticle(0)
        # ... and add a dummy/ghost atom next to it
        nonbonded.addParticle(0, 1, 0)
        # ... that won't interact with the restrained atom
        nonbonded.addException(i, j, 0, 1, 0)
        # ... but will be have a harmonic restraint ('bond')
        # between the two atoms
        restraint.addBond(i, j, 0*nanometers,
                          RESTRAINT_K*kilocalories_per_mole/angstrom**2)
        dummyIndex.append(j)
        input_positions.append(positions[i])

    integrator = LangevinIntegrator(temperature*kelvin, FRICTION/picosecond,
                                    TIMESTEP_PS*picoseconds)
    sim = make_simulation(parm.topology, system, integrator,
                          platform, properties)
    sim.context.setPositions(input_positions)
    sim.context.setVelocitiesToTemperature(temperature*kelvin)

    steps = int(round(equil_time / TIMESTEP_PS))
    print(f"  Equilibrating for {equil_time} ps ({steps} steps), "
          f"{len(solute_heavy_atom_idx)} restrained solute heavy atoms ...")
    sim.step(steps)

    all_positions = sim.context.getState(
        getPositions=True, enforcePeriodicBox=True).getPositions()
    # we don't want to write the dummy atoms, so we only
    # write the positions of atoms up to the first dummy atom index
    relevant_positions = all_positions[:dummyIndex[0]]
    out_file = os.path.join(out_dir, eq_file_name)
    with open(out_file, 'w') as handle:
        PDBFile.writeFile(sim.topology, relevant_positions, handle)

    return None


def produce(out_dir, idx, lig_resname, prot_resname, eq_pdb, parm, parm_file,
            set_hill_height, sim_time, temperature,
            platform=None, properties=None):
    """An OpenBPMD production simulation function. Peptide RMSD is biased
    with well-tempered metadynamics. The integrator uses a 2 fs time step
    with hydrogen mass repartitioning, runs for '--sim_time' ns by default
    and writes 100 frames.

    Writes a 'trj.dcd', 'COLVAR.npy', 'bias_*.npy' and 'sim_log.csv' files
    during the metadynamics simulation in the '{out_dir}/rep_{idx}' directory.

    Parameters
    ----------
    out_dir : str
        Directory where your equilibration PDBs and 'rep_*' dirs are at.
    idx : int
        Current replica index.
    lig_resname : str
        Selection string for the peptide.
    prot_resname : str
        Selection string for the receptor.
    eq_pdb : str
        Name of the PDB for equilibrated system.
    parm : OpenMM parameter file object
        Used to create the OpenMM System object.
    parm_file : str
        The name of the parameter or topology file of the system.
    set_hill_height : float
        Metadynamic hill height, in kcal/mol.
    sim_time : float
        Length of the replica, in ns.
    temperature : float
        Simulation temperature, in K.
    platform, properties :
        Optional OpenMM platform and platform properties.
    """
    # First, assign the replica directory to which we'll write the files
    write_dir = os.path.join(out_dir, f'rep_{idx}')
    # Get the anchor atoms by ...
    universe = mda.Universe(eq_pdb,
                            format='XPDB', in_memory=True)
    check_selection(universe, lig_resname, "peptide")
    check_selection(universe, prot_resname, "receptor")
    # ... finding the protein's COM ...
    prot_com = universe.select_atoms(prot_resname).center_of_mass()
    x, y, z = prot_com[0], prot_com[1], prot_com[2]
    # ... and taking the heavy backbone atoms within 5A of the COM
    sel_str = f'point {x} {y} {z} 5 and backbone and not name H*'
    anchor_atoms = universe.select_atoms(sel_str)
    # ... or 10 angstrom
    if len(anchor_atoms) == 0:
        sel_str = f'point {x} {y} {z} 10 and backbone and not name H*'
        anchor_atoms = universe.select_atoms(sel_str)
    if len(anchor_atoms) == 0:
        raise ValueError(
            "No anchor atoms found within 10 A of the receptor centre of "
            "mass; cannot define the alignment group for the RMSD CV."
        )

    anchor_atom_idx = anchor_atoms.indices.tolist()

    # Get indices of peptide heavy atoms
    lig = universe.select_atoms(f'{lig_resname} and not name H*')
    lig_ha_idx = lig.indices.tolist()

    system = parm.createSystem(nonbondedMethod=PME,
                               nonbondedCutoff=1*nanometers,
                               constraints=HBonds, hydrogenMass=4*amu)
    # get the atom positions for the system from the equilibrated
    # system
    input_positions = PDBFile(eq_pdb).getPositions()

    # Add an 'empty' flat-bottom restraint to fix the issue with PBC.
    # Without one, RMSDForce object fails to account for PBC.
    k = 0*kilojoules_per_mole  # NOTE - 0 kJ/mol constant
    upper_wall = 10.00*nanometer
    fb_eq = '(k/2)*max(distance(g1,g2) - upper_wall, 0)^2'
    upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
    upper_wall_rest.addGroup(lig_ha_idx)
    upper_wall_rest.addGroup(anchor_atom_idx)
    upper_wall_rest.addBond([0, 1])
    upper_wall_rest.addGlobalParameter('k', k)
    upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
    system.addForce(upper_wall_rest)

    alignment_indices = lig_ha_idx + anchor_atom_idx

    rmsd = RMSDForce(input_positions, alignment_indices)
    # Set up the typical metadynamics parameters
    hill_height = set_hill_height*kilocalories_per_mole
    hill_width = HILL_WIDTH  # nm, also known as sigma

    grid_width = hill_width / 5
    # 'grid' here refers to the number of grid points
    grid = int(abs(GRID_MIN - GRID_MAX) / grid_width)

    rmsd_cv = BiasVariable(rmsd, GRID_MIN, GRID_MAX, hill_width,
                           False, gridWidth=grid)

    steps = int(round(sim_time * 1000 / TIMESTEP_PS))
    # Keep the step count a whole number of CV-recording chunks
    steps = max(COLVAR_CHUNK, (steps // COLVAR_CHUNK) * COLVAR_CHUNK)
    # Write N_FRAMES frames, on a multiple of the bias deposition period
    report_interval = max(
        BIAS_FREQUENCY,
        (steps // N_FRAMES // BIAS_FREQUENCY) * BIAS_FREQUENCY)
    # Bias is saved every ns, or at the end of a shorter run
    save_frequency = min(
        500000, max(BIAS_FREQUENCY, (steps // BIAS_FREQUENCY) * BIAS_FREQUENCY))

    # Stale bias files would be read back in and added to the new run.
    stale_bias = glob.glob(os.path.join(write_dir, 'bias_*.npy'))
    if stale_bias:
        print(f"  WARNING: {len(stale_bias)} bias file(s) already in "
              f"{write_dir}. OpenMM will load them, so this replica "
              f"continues the earlier biased run instead of starting "
              f"fresh. Delete the directory for a clean replica.")

    # define the metadynamics object
    # deposit bias every 250 steps (0.5 ps), BF = 4, write bias every ns
    meta = Metadynamics(system, [rmsd_cv], temperature*kelvin, BIAS_FACTOR,
                        hill_height, BIAS_FREQUENCY, biasDir=write_dir,
                        saveFrequency=save_frequency)

    # Set up and run metadynamics
    integrator = LangevinIntegrator(temperature*kelvin, FRICTION/picosecond,
                                    TIMESTEP_PS*picoseconds)

    simulation = make_simulation(parm.topology, system, integrator,
                                 platform, properties)
    simulation.context.setPositions(input_positions)
    simulation.context.setVelocitiesToTemperature(temperature*kelvin)

    trj_name = os.path.join(write_dir, 'trj.dcd')

    simulation.reporters.append(DCDReporter(trj_name, report_interval))
    simulation.reporters.append(StateDataReporter(stdout, report_interval,
                                step=True, speed=True, progress=True,
                                totalSteps=steps, remainingTime=True,
                                separator='\t\t'))
    simulation.reporters.append(StateDataReporter(
                                os.path.join(write_dir, 'sim_log.csv'),
                                report_interval,
                                step=True, temperature=True, progress=True,
                                remainingTime=True, speed=True,
                                totalSteps=steps, separator=','))

    print(f"\n> Simulating {steps} steps ({sim_time} ns), writing a frame "
          f"every {report_interval} steps ...")

    colvar_array = np.array([meta.getCollectiveVariables(simulation)])
    for i in range(0, int(steps), COLVAR_CHUNK):
        if (i // COLVAR_CHUNK) % 100 == 0:
            # periodically flush the stored COLVAR to disk
            np.save(os.path.join(write_dir, 'COLVAR.npy'), colvar_array)
        meta.step(simulation, COLVAR_CHUNK)
        current_cvs = meta.getCollectiveVariables(simulation)
        # record the CVs every 1 ps
        colvar_array = np.append(colvar_array, [current_cvs], axis=0)
    np.save(os.path.join(write_dir, 'COLVAR.npy'), colvar_array)

    # center everything using MDTraj, to fix any PBC imaging issues
    mdu = md.load(trj_name, top=parm_file)
    mdu.image_molecules()
    mdu.save(trj_name)

    return None


# ======================== PER-COMPLEX DRIVER ========================
def run_complex(args, name, complex_dir):
    """Run the whole OpenBPMD protocol for one peptide + protein complex."""
    print(f"\n{'='*60}")
    print(f"Complex: {name}")
    print(f"{'='*60}")

    parameters = os.path.join(complex_dir, f"{name}{args.prmtop_suffix}")
    structure = os.path.join(complex_dir, f"{name}{args.inpcrd_suffix}")
    pdb_ed = os.path.join(complex_dir, f"{name}{args.pdb_suffix}")

    for path, what in [(parameters, "topology"), (structure, "coordinates"),
                       (pdb_ed, "solute PDB")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{what} file not found: {path}. Run build_topology.py "
                f"first, or adjust the --*_suffix options."
            )
    # mdtraj identifies the topology format from the file extension
    if not parameters.endswith(('.prmtop', '.parm7', '.prm7')):
        raise ValueError(
            f"MDTraj cannot read '{parameters}': the AMBER topology needs a "
            f".prmtop, .parm7 or .prm7 extension."
        )

    output = os.path.join(complex_dir, args.output)
    os.makedirs(output, exist_ok=True)

    # --- work out which residues are the peptide and which the receptor ---
    ranges = chain_ranges_from_pdb(pdb_ed)
    print(f"  Chains detected in {os.path.basename(pdb_ed)}: {len(ranges)}")
    for i, (start, end) in enumerate(ranges, 1):
        print(f"    chain {i}: residues {start}-{end} "
              f"({end - start + 1} residues)")
    if len(ranges) < max(args.lig_chain, args.prot_chain):
        raise ValueError(
            f"Asked for peptide chain {args.lig_chain} and receptor chain "
            f"{args.prot_chain}, but only {len(ranges)} chain(s) were found "
            f"in {pdb_ed}. If the file has no TER records, regenerate it "
            f"with build_topology.py."
        )

    lig_start, lig_end = ranges[args.lig_chain - 1]
    prot_start, prot_end = ranges[args.prot_chain - 1]
    print(f"  Peptide  = chain {args.lig_chain} "
          f"(residues {lig_start}-{lig_end})")
    print(f"  Receptor = chain {args.prot_chain} "
          f"(residues {prot_start}-{prot_end})")

    coords = AmberInpcrdFile(structure)
    parm = AmberPrmtopFile(parameters)
    platform, properties = get_platform(args)
    if platform is not None:
        print(f"  OpenMM platform: {platform.getName()} {properties}")

    # Minimize
    min_file_name = 'minimized_system.pdb'
    min_pdb = os.path.join(output, min_file_name)
    if not os.path.isfile(min_pdb):
        print("Minimizing...")
        minimize(parm, coords.positions, output, min_file_name,
                 platform, properties)

    # Equilibrate
    eq_file_name = 'equil_system.pdb'
    eq_pdb = os.path.join(output, eq_file_name)
    if not os.path.isfile(eq_pdb):
        print("Equilibrating...")
        equilibrate(min_pdb, parm, output, eq_file_name, args.equil_time,
                    args.temperature, platform, properties)
    cent_eq_pdb = os.path.join(output, 'centred_' + eq_file_name)
    if os.path.isfile(eq_pdb) and not os.path.isfile(cent_eq_pdb):
        mdu = md.load(eq_pdb, top=parameters)
        mdu.image_molecules()
        mdu.save_pdb(cent_eq_pdb)

    # The equilibrated PDB is written by OpenMM, which numbers residues
    # from 1, while the centred PDB is written by MDTraj, which numbers
    # them from 0. The analysis selections therefore need an offset.
    offset = first_residue_number(cent_eq_pdb) - first_residue_number(eq_pdb)
    if offset != 0:
        print(f"  Residue numbering offset between the equilibrated and "
              f"centred PDB: {offset}")

    # Selections against the equilibrated PDB (used to set up the CV) ...
    lig_resname = f"protein and resid {lig_start}:{lig_end}"
    prot_resname = f"protein and resid {prot_start}:{prot_end}"
    # ... and against the centred PDB (used for the analysis)
    lig_resname_analysis = f"resid {lig_start + offset}:{lig_end + offset}"
    lig_resname_score = f"protein and {lig_resname_analysis}"
    prot_resname_analysis = (f"protein and resid "
                             f"{prot_start + offset}:{prot_end + offset}")

    # Record what was actually run, for provenance
    with open(os.path.join(complex_dir, 'output_info.txt'), 'w') as f:
        f.write(f"lig_resname: {lig_resname}\n")
        f.write(f"lig_resname_analysis: {lig_resname_analysis}\n")
        f.write(f"lig_resname_score: {lig_resname_score}\n")
        f.write(f"prot_resname: {prot_resname}\n")
        f.write(f"prot_resname_analysis: {prot_resname_analysis}\n")
        f.write(f"parameters: {parameters}\n")
        f.write(f"structure: {structure}\n")
        f.write(f"nreps: {args.nreps}\n")
        f.write(f"hill_height: {args.hill_height} kcal/mol\n")
        f.write(f"sim_time: {args.sim_time} ns\n")
        f.write(f"equil_time: {args.equil_time} ps\n")
        f.write(f"temperature: {args.temperature} K\n")

    # Run NREPS number of production simulations
    for idx in range(1, (args.nreps+1)):
        rep_dir = os.path.join(output, f'rep_{idx}')
        os.makedirs(rep_dir, exist_ok=True)
        if os.path.isfile(os.path.join(rep_dir, 'bpm_results.csv')):
            print(f"Replica {idx} already scored -- skipping")
            continue
        print(f"Running BPMD simulation {idx} of {args.nreps}")

        produce(output, idx, lig_resname, prot_resname, eq_pdb, parm,
                parameters, args.hill_height, args.sim_time,
                args.temperature, platform, properties)

        trj_name = os.path.join(rep_dir, 'trj.dcd')

        PoseScoreArr = get_pose_score(cent_eq_pdb, trj_name,
                                      lig_resname_score)

        ContactScoreArr = get_contact_score(cent_eq_pdb, trj_name,
                                            lig_resname_score,
                                            lig_resname_analysis,
                                            prot_resname_analysis,
                                            args.sim_time)

        # Calculate the CompScore at every frame
        n_scored = min(len(PoseScoreArr), len(ContactScoreArr))
        if len(PoseScoreArr) != len(ContactScoreArr):
            print(f"  WARNING: {len(PoseScoreArr)} PoseScores but "
                  f"{len(ContactScoreArr)} ContactScores; using the first "
                  f"{n_scored} frames of each")
        PoseScoreArr = PoseScoreArr[:n_scored]
        ContactScoreArr = ContactScoreArr[:n_scored]
        CompScoreArr = PoseScoreArr - COMP_CONTACT_WEIGHT * ContactScoreArr

        Scores = np.stack((CompScoreArr, PoseScoreArr, ContactScoreArr),
                          axis=-1)

        # Save a DataFrame to CSV
        df = pd.DataFrame(Scores, columns=['CompScore', 'PoseScore',
                                           'ContactScore'])
        df.to_csv(os.path.join(rep_dir, 'bpm_results.csv'), index=False)

    collect_results(output, output, args.sim_time)

    return None


def discover_complexes(args):
    """Find the complexes to run inside --work_dir."""
    if args.complex:
        names = args.complex
    else:
        names = sorted(
            entry for entry in os.listdir(args.work_dir)
            if os.path.isdir(os.path.join(args.work_dir, entry))
            and os.path.isfile(os.path.join(
                args.work_dir, entry, f"{entry}{args.prmtop_suffix}"))
        )
    return [(name, os.path.join(args.work_dir, name)) for name in names]


def main():
    args = parse_arguments()

    complexes = discover_complexes(args)
    if not complexes:
        print(f"No complexes found in '{args.work_dir}'.")
        print(f"Expected '<work_dir>/<name>/<name>{args.prmtop_suffix}', "
              f"as produced by build_topology.py.")
        return 1

    print(f"Found {len(complexes)} complex(es): "
          f"{', '.join(n for n, _ in complexes)}")

    failures = []
    for name, complex_dir in complexes:
        try:
            run_complex(args, name, complex_dir)
        except Exception as exc:
            failures.append((name, exc))
            print(f"ERROR while running '{name}': {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} complex(es) failed:", file=sys.stderr)
        for name, exc in failures:
            print(f"  {name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
