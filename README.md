# BPMD-peptides

**Open Binding Pose Metadynamics (OpenBPMD) for peptide–protein complexes.**

Docking a peptide onto a receptor gives you a pose. It does not tell you
whether that pose is *stable*. This repository answers that question with
binding pose metadynamics: a small, well-tempered metadynamics bias is
applied to the peptide's RMSD, which actively pushes the peptide out of its
starting pose. A peptide that is genuinely well bound resists; a peptide that
is only superficially docked slides away and loses its contacts.

The implementation follows binding pose metadynamics as introduced by
[Clark et al. (2016)](https://doi.org/10.1021/acs.jctc.6b00201) and the
open-source OpenMM implementation (OpenBPMD), adapted here for **peptide
ligands** — where the "ligand" is a whole protein chain rather than a small
molecule, so it is selected by chain instead of by residue name.

## Run it on Google Colab — no GPU needed

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/BPMD-peptides/blob/main/BPMD_peptides_Colab.ipynb)

If you do not have a GPU or a cluster, open
[`BPMD_peptides_Colab.ipynb`](BPMD_peptides_Colab.ipynb) in Colab, switch the
runtime to a T4 GPU and work through the cells. It installs everything,
fetches or accepts your structure, runs both stages and plots the scores.

The notebook computes on Colab's local disk and copies results to Google
Drive after the topology is built and after every replica. Drive is a network
mount that drops out under sustained writes (`Transport endpoint is not
connected`), so it is used as the place results are saved to, not the place
they are computed in.

Mount Drive when the notebook asks. Colab sessions are shorter than a full
10 × 10 ns job, but each finished replica is banked as it completes and
re-running the simulation cell resumes from there, so a long job can be
finished across several sessions. Start with one short replica to check your
system and measure the speed before committing to the full protocol.

---

Two scripts, run in order:

| Stage | Script | What it does |
|---|---|---|
| 1 | `build_topology.py` | PDB complex → solvated AMBER topology (pdb2pqr/PROPKA for protonation states, `tleap` for parameters) |
| 2 | `run_bpmd.py` | minimise → equilibrate → *N* metadynamics replicas → pose stability scores |

> These were previously called `tleap_vParser_v4.py` and `peptides_BPMD.py`.
> The pipeline is unchanged; several bugs were fixed and both scripts now have
> a full command-line interface. See [What changed](#what-changed-from-the-original-scripts).

---

## Table of contents

- [Run it on Google Colab](#run-it-on-google-colab--no-gpu-needed)
- [The scores](#the-scores)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Preparing your input PDB](#preparing-your-input-pdb)
- [Stage 1 — `build_topology.py`](#stage-1--build_topologypy)
- [Stage 2 — `run_bpmd.py`](#stage-2--run_bpmdpy)
- [Output files](#output-files)
- [Interpreting the results](#interpreting-the-results)
- [Restarting and resuming](#restarting-and-resuming)
- [Troubleshooting](#troubleshooting)
- [What changed from the original scripts](#what-changed-from-the-original-scripts)
- [Citing](#citing)
- [License](#license)

---

## The scores

Each replica is scored frame by frame, and the reported value is the average
over the **last 2 ns** of every replica.

| Score | Definition | Reads as |
|---|---|---|
| **PoseScore** | RMSD (Å) of the peptide's heavy atoms, after aligning the receptor backbone, relative to the first production frame | How far the peptide moved. Lower = pose retained |
| **ContactScore** | Number of peptide–receptor heavy-atom contacts within 3.5 Å, normalised by the average over the first 1 ns and capped at 1 | What fraction of the original interface survives. Higher = contacts retained |
| **CompScore** | `PoseScore − 5 × ContactScore` | Single number for ranking poses. Lower = more stable |

The composite score deliberately rewards keeping contacts and punishes
drifting away, so it separates a peptide that merely rearranges locally from
one that leaves the site.

---

## Installation

*(Skip this if you are using the [Colab notebook](#run-it-on-google-colab--no-gpu-needed),
which installs everything for you.)*

Both stages need a mix of AMBER tooling and Python MD libraries, which is
easiest through conda. The environment below is the one this pipeline is
tested with:

```bash
git clone https://github.com/pablo-arantes/BPMD-peptides.git
cd BPMD-peptides

conda env create -f environment.yml
conda activate bpmd-peptides
```

Or add the pieces to an environment you already have:

```bash
conda install -c conda-forge ambertools openmm pymol-open-source mdtraj \
                            "numpy<2" pandas scipy netcdf4
pip install MDAnalysis biopandas pdb2pqr
```

### What each dependency is for

| Package | Used by | Why |
|---|---|---|
| `ambertools` (`tleap`) | stage 1 | force-field assignment, solvation, ions |
| `pdb2pqr` + `propka` | stage 1 | pH-dependent protonation states |
| `pymol-open-source` | stage 1 | structural cleanup between pdb2pqr and tleap |
| `biopandas` | stage 1 | residue renaming, stripping waters/hydrogens |
| `openmm` | stage 2 | minimisation, equilibration, metadynamics |
| `mdtraj` | stage 2 | PBC imaging of the trajectories |
| `MDAnalysis` | stage 2 | RMSD and contact analysis |
| `pandas`, `numpy` | both | tabular output and scoring |

### Checking the install

```bash
which tleap pdb2pqr30            # stage 1 executables must be on PATH
python -m openmm.testInstallation # lists the available OpenMM platforms
```

### GPU

Stage 2 is the expensive part and should run on a GPU. On Linux with an
NVIDIA card, install a CUDA-enabled OpenMM:

```bash
conda install -c conda-forge openmm cuda-version=12
```

then pass `--platform CUDA`. Without `--platform`, OpenMM silently picks the
fastest platform it can find, which may be the CPU — check the speed printed
in the progress output if a run seems slow.

---

## Quick start

```bash
# A directory with one PDB per complex
ls
#  complex1.pdb  complex2.pdb

# Stage 1: build the AMBER topologies
python build_topology.py --ff ff14SB --ph 7.4 --ssbond_complex no

# Stage 2: 10 metadynamics replicas per complex.
# --peptide_chain says which chain is the peptide; it is required.
python run_bpmd.py --peptide_chain 1 --nreps 10 --platform CUDA

# Final scores
cat complex1/peptide_pose/results.csv
```

Stage 1 creates one directory per complex and stage 2 works inside it, so both
scripts can be pointed at a whole batch of complexes at once.

---

## Preparing your input PDB

This matters more than any option below — most failures come from the input
structure rather than the code.

**1. One PDB file per complex**, containing the peptide *and* the receptor,
already in the pose you want to test (from docking, a crystal structure, or a
model).

**2. You tell the script which chain is the peptide.** Chains are numbered in
the order they appear in the PDB, split on `TER` records and on changes of
chain ID:

```
ATOM  ...  peptide chain  (chain P)   <- chain 1
TER
ATOM  ...  receptor chain (chain A)   <- chain 2
TER
END
```

`--peptide_chain` is **required**, because picking the wrong molecule produces
perfectly plausible numbers for the wrong thing rather than an error. There is
no default to get silently wrong:

```bash
python run_bpmd.py --peptide_chain 1          # peptide first in the file
python run_bpmd.py --peptide_chain 2          # receptor first in the file
```

`--receptor_chain` only needs to be given when the complex has **more than two
chains** — with exactly two, the receptor is the other one and the script says
which it picked. For a receptor that is itself a dimer, or a complex with
cofactor chains, be explicit:

```bash
python run_bpmd.py --peptide_chain 3 --receptor_chain 1
```

Stage 2 prints every chain it detected, with its residue range, and then the
peptide and receptor it assigned — all before any simulation starts. **Check
those lines on the first run.** If you do not know the numbering yet, run it
once with any value and read the list off the output.

**3. Keep the header `SSBOND` records** if the structure has disulfides.
They are read from the *original* PDB, because pdb2pqr discards header lines.

**4. Clean the structure first.** Remove alternate conformations, ligands and
cofactors you do not want parameterised, and model in any missing loops.
`tleap` bonds consecutive residues of a chain regardless of distance, so a
gap in the middle of a chain becomes a stretched bond — stage 1 warns when it
sees one, but it cannot fix it.

**5. Crystallographic waters are removed** automatically, as are hydrogens
(`tleap` adds its own).

---

## Stage 1 — `build_topology.py`

```bash
python build_topology.py [options]
```

Per PDB file found in `--input_dir`:

1. `SSBOND` records are parsed from the original PDB header.
2. **pdb2pqr + PROPKA** assign protonation states at the requested pH.
3. Residue names are edited: inter-chain `CYX`→`CYS` when inter-chain
   disulfides are disabled, plus any user ionization overrides.
4. **PyMOL** cleans up the structure.
5. **BioPandas** strips waters, `OXT` and hydrogens, keeping `TER` records.
6. **tleap** runs twice — once without ions to measure the box volume, then
   again to neutralise, add salt at the requested concentration, and write the
   topology.

### Options

| Option | Default | Description |
|---|---|---|
| `--ff {ff14SB,ff19SB}` | `ff14SB` | Protein force field. `ff14SB` pairs with TIP3P water, `ff19SB` with OPC |
| `--size_box` | `10.0` | Buffer between solute and the edge of the truncated-octahedral box (Å) |
| `--concentration` | `0.15` | NaCl concentration added *on top of* neutralisation (M) |
| `--ph` | `7.0` | pH used by PROPKA to assign titration states |
| `--titration_method {propka,none}` | `propka` | `none` uses force-field default states and ignores `--ph` |
| `--ssbond_complex {yes,no}` | `no` | Whether to form **inter-chain** disulfides. Intra-chain disulfides are always formed |
| `--ion_aminoacid_list` | none | File of manual protonation-state overrides (see below) |
| `--input_dir` | `./` | Directory scanned for `.pdb` files |
| `--work_dir` | `./` | Where the per-complex output directories are created |
| `--pdb` | all | Process only these PDB files |
| `--pdb2pqr_exe`, `--tleap_exe` | auto | Override executable paths |

### Disulfide bonds

`--ssbond_complex` controls only **inter-chain** bonds, i.e. a covalent link
between your peptide and the receptor:

- `no` (default) — the peptide is a non-covalent ligand. Inter-chain
  disulfides are dropped and the cysteines involved are reverted from `CYX`
  to free `CYS`. This is what you want for pose stability: a covalently
  tethered peptide cannot leave, so the scores would be meaningless.
- `yes` — inter-chain disulfides are formed. Use this for genuinely covalent
  complexes (for example the two chains of insulin).

Intra-chain disulfides are always formed, in both modes.

### Manual ionization overrides

PROPKA's assignment can be overridden per residue. Residue numbers refer to
the **original PDB numbering**, and are translated automatically if pdb2pqr
renumbers anything:

```
# RESNAME  RESID  [CHAIN]
GLH   35    A      # protonated (neutral) Glu35 of chain A
HIP   64           # doubly protonated His64 in every chain
ASH   102   B
LYN   15    A      # neutral lysine
```

Overrides are validated against the residue's ionization family, so a typo
cannot turn a valine into a glutamate — incompatible requests are reported
and skipped. Available states: `GLU/GLH`, `ASP/ASH`, `HIS/HID/HIE/HIP`,
`CYS/CYX`, `LYS/LYN`. An example file is in
[`examples/ionization_aminoacid.txt`](examples/ionization_aminoacid.txt).

---

## Stage 2 — `run_bpmd.py`

```bash
python run_bpmd.py [options]
```

For each complex found in `--work_dir`:

1. **Minimisation** of the solvated system.
2. **Restrained NVT equilibration** (1 ns by default, 5 kcal/mol/Å² on solute
   heavy atoms, restrained to massless dummy particles so a barostat could not
   drag them).
3. **`--nreps` production replicas**, each 10 ns of well-tempered metadynamics
   at 300 K with a 2 fs timestep and hydrogen mass repartitioning, biasing the
   peptide RMSD (hill height 0.5 kcal/mol, σ = 0.02 Å, bias factor 4,
   deposited every 0.5 ps). 100 frames are written per replica.
4. **Scoring** of each replica, then the averages across replicas.

The collective variable is the RMSD of the peptide's heavy atoms together with
receptor backbone anchor atoms near the receptor's centre of mass, which keeps
the RMSD well defined under periodic boundary conditions.

### Options

| Option | Default | Description |
|---|---|---|
| `--work_dir` | `./` | Directory holding the per-complex directories from stage 1 |
| `--complex` | all | Names of the complexes to run |
| `--output` | `peptide_pose` | Name of the output directory created inside each complex directory |
| `--peptide_chain` | **required** | 1-based index of the chain that is the **peptide** — the molecule that is biased and scored |
| `--receptor_chain` | inferred | 1-based index of the chain that is the **receptor**. Optional only for a two-chain complex |
| `--nreps` | `10` | Number of replicas, run in series |
| `--hill_height` | `0.5` | Metadynamics hill height (kcal/mol) |
| `--sim_time` | `10.0` | Length of each replica (ns) |
| `--equil_time` | `1000.0` | Restrained equilibration (ps) |
| `--temperature` | `300.0` | Temperature (K) |
| `--platform` | auto | `CUDA`, `OpenCL`, `HIP`, `CPU` or `Reference` |
| `--precision` | `mixed` | Precision for GPU platforms |
| `--device_index` | auto | GPU index, e.g. `0` or `0,1` |
| `--prmtop_suffix`, `--inpcrd_suffix`, `--pdb_suffix` | `_amber.prmtop`, `_amber.inpcrd`, `_ed.pdb` | File-naming conventions from stage 1 |

Replicas differ only in their random velocities and the stochastic
trajectory, which is the point: with 10 replicas you get a spread rather than
a single number, and `results_per_replica.csv` lets you see whether they
agree.

### Cost

Ten 10 ns replicas is 100 ns of metadynamics per complex, plus 1 ns of
equilibration. On a modern GPU this is a few hours per complex. Scale down
with `--nreps` and `--sim_time` while setting things up — but note the final
scores average the **last 2 ns**, so runs shorter than that average whatever
is available and are only useful as smoke tests.

---

## Output files

```
complex1.pdb                              your input
complex1/
├── complex1_pka.pdb/.pqr/.log            pdb2pqr output (protonation states)
├── complex1_pymol.pdb                    after PyMOL cleanup
├── complex1_ed.pdb                       dry solute, TER records, fed to tleap
├── complex1_no_ions_tleap.in             first tleap pass (box volume only)
├── complex1_no_ions.log                  its log — the box volume is read here
├── complex1_no_ions.prmtop/.inpcrd/.pdb  intermediate, not used further
├── complex1_with_ions_tleap.in           second tleap pass
├── complex1_with_ions.log                its log — check this if tleap fails
├── complex1_amber.prmtop                 AMBER topology    <- stage 2 input
├── complex1_amber.inpcrd                 AMBER coordinates <- stage 2 input
├── complex1_amber.pdb                    solvated system as PDB
├── output_info.txt                       the selections and settings used
└── peptide_pose/
    ├── minimized_system.pdb
    ├── equil_system.pdb                 end of restrained equilibration
    ├── centred_equil_system.pdb         PBC-imaged, topology for the analysis
    ├── rep_1/
    │   ├── trj.dcd                      100 frames
    │   ├── COLVAR.npy                   RMSD CV every 1 ps
    │   ├── bias_*.npy                   accumulated metadynamics bias
    │   ├── sim_log.csv                  step, temperature, speed
    │   └── bpm_results.csv              per-frame CompScore/PoseScore/ContactScore
    ├── rep_2/ ... rep_N/
    ├── results_per_replica.csv          one row per replica
    └── results.csv                      FINAL SCORES
```

`output_info.txt` records the selections that were actually used, which is the
first place to look if a result seems wrong:

```
lig_resname: protein and resid 1:21            # peptide, for the CV
lig_resname_analysis: resid 0:20               # peptide, for the analysis
lig_resname_score: protein and resid 0:20
prot_resname: protein and resid 22:51          # receptor, for the CV
prot_resname_analysis: protein and resid 21:50
```

The analysis selections are offset by one because MDTraj numbers residues
from 0 while OpenMM numbers them from 1. The offset is measured from the
files at runtime, not assumed.

`results.csv` holds the mean and standard deviation of each score over the
last 2 ns of all replicas, plus the number of replicas included:

```csv
CompScore,CompScoreSD,PoseScore,PoseScoreSD,ContactScore,ContactScoreSD,nreps
-3.142,0.871,1.573,0.42,0.943,0.061,10
```

---

## Interpreting the results

The scores are most useful **comparatively** — ranking poses of the same
peptide, or comparing variants of a peptide against the same receptor. Taken
on its own, a rough reading is:

- **PoseScore ≲ 2 Å with ContactScore ≳ 0.7** — the peptide held its pose
  against the bias. A stable binding mode.
- **PoseScore ≫ 2 Å with a collapsing ContactScore** — the bias pushed the
  peptide out and it did not come back. The pose is not stable.
- **High PoseScore but ContactScore still near 1** — the peptide rearranged
  but stayed on the interface. Often a sign that the binding site is right but
  the modelled pose is not; worth inspecting the trajectory.
- **A large spread across replicas** (check `results_per_replica.csv`) means
  the result is not converged. Run more replicas before trusting the ranking.

Always look at a trajectory before drawing a conclusion. The PoseScore cannot
distinguish a peptide that unbinds from one that slides along a groove, but
you can see the difference immediately:

```bash
# in the peptide_pose directory
pymol centred_equil_system.pdb rep_1/trj.dcd
```

---

## Restarting and resuming

Both stages are restartable, and both skip work that is already on disk:

- Stage 1 overwrites its outputs, so re-running it is always safe.
- Stage 2 reuses `minimized_system.pdb` and `equil_system.pdb` if they exist,
  and skips any replica that already has a `bpm_results.csv`.

When re-running to pick up extra replicas, **pass the same `--sim_time` you
originally used.** The scoring window is expressed in nanoseconds, so the
script works out "the last 2 ns" from the replica length you give it — tell it
10 ns when the replicas are 5 ns long and it will average the wrong window.
The value used is recorded in `output_info.txt`.

To rerun a replica, **delete its whole directory** rather than individual
files. An interrupted replica leaves `bias_*.npy` files behind, and OpenMM
reads those back in on the next start, so the new run would continue the old
biased simulation instead of starting fresh. Stage 2 warns when it finds them.

---

## Troubleshooting

**`Could not find pdb2pqr on PATH`** — the conda environment is not active, or
`pdb2pqr` is installed somewhere else. Pass `--pdb2pqr_exe /path/to/pdb2pqr30`.

**`--peptide_chain N was requested, but only 1 chain(s) were found`** — the
`_ed.pdb` has no chain separation. Regenerate it with
`build_topology.py`; if your input PDB genuinely has a single chain ID for
both molecules, add a `TER` between them or give them distinct chain IDs.

**`The peptide ... selection matches no atoms`** — `--peptide_chain` /
`--receptor_chain` point at the wrong chain, or the residues are not recognised
as protein by MDAnalysis. Non-standard residues and the `NHE` terminal cap are
not covered by MDAnalysis's `protein` keyword.

**`MDTraj cannot read ...`** — the AMBER topology must end in `.prmtop`,
`.parm7` or `.prm7`. This is why stage 1 writes `_amber.prmtop` rather than
`.top`.

**`tleap reported a fatal error`** — read the `*_tleap.log` file in the
complex directory. Common causes: a non-standard residue with no parameters,
a missing atom in the input structure, or an unrecognised HETATM group.

**Unusually long bonds or an exploding minimisation** — check the stage 1 log
for the residue-numbering-gap warning. A chain with a missing loop gets bonded
across the gap.

**The simulation is far slower than expected** — OpenMM chose the CPU
platform. Pass `--platform CUDA` and check `python -m openmm.testInstallation`.

---

## What changed from the original scripts

The pipeline, protocol and scoring are unchanged. These are bug fixes and
usability changes; the ones that **change your results** are marked.

### Stage 1

1. **`TER` records were being lost** *(changes results)* — BioPandas keeps
   `TER` lines in a separate dataframe, so writing only `ATOM`/`HETATM`
   dropped every chain terminator. `tleap` then bonded the last residue of one
   chain to the first residue of the next: on a test complex this produced a
   **34.6 Å "peptide bond"** fusing peptide to receptor, with uncapped termini.
   `TER` records are now restored, and the resulting topology was verified to
   have no such bond and properly capped termini.
2. **`--ph` had no effect** *(changes results)* — pdb2pqr silently ignores
   `--with-ph` unless a titration-state method is requested. Verified: pH 2 and
   pH 12 gave *identical* protonation states. PROPKA is now requested
   explicitly; `--titration_method none` restores the old behaviour.
3. **Topology files are now `.prmtop`/`.inpcrd`** instead of `.top`/`.crd`,
   because MDTraj in stage 2 identifies the format from the extension and
   refuses `.top`. The two stages now connect without renaming files by hand.
4. **Disulfide residue numbers are mapped** from the original PDB numbering
   through to the final file before being written as `tleap` bond commands, so
   disulfides are not silently dropped if pdb2pqr renumbers residues.
5. `pdb2pqr` is located automatically (`pdb2pqr30`, then `pdb2pqr`).
6. `tleap` failures are detected — it often exits successfully after a fatal
   error, so the log is now checked and the expected output files verified.
7. Subprocesses are called without a shell, so paths with spaces work.
8. `retain_order` is set *before* PyMOL loads the structure, where it applies.
9. Hydrogen removal falls back to atom names when the PDB element column is
   blank.
10. `SSBOND` parsing uses the fixed PDB columns, with token splitting as a
    fallback.
11. Unused imports of `openmm`, `pdbfixer` and `openbabel` were removed —
    they were never called, and `openbabel` in particular is awkward to install.
12. Added `--input_dir`, `--work_dir`, `--pdb`, residue-gap warnings, and a
    summary of successes and failures.

### Stage 2

1. **A real command-line interface** — `argparse` was imported but unused, and
   the working directory was hard-coded to `/content/` (a Colab path). Every
   setting that was a module-level variable is now an option.
2. **The peptide chain is named explicitly** via the required
   `--peptide_chain`, instead of the peptide being assumed to be the first
   chain in the file. `--receptor_chain` is inferred for a two-chain complex
   and required otherwise.
3. **Chain detection no longer depends on `TER` alone**, and the fragile
   `pdb_file.split('.')[0]` / `name.split('/')[2]` path handling is gone — the
   latter only worked for paths of one exact depth.
4. **The `-1` residue offset is detected, not assumed** — the analysis
   selections need it because MDTraj numbers residues from 0 while OpenMM
   numbers them from 1 (both verified). It is now measured from the files, so
   it cannot silently become wrong.
5. **`not name *H*` → `not name H*`** in the contact selection
   *(changes results slightly)* — the glob `*H*` also excluded heavy atoms
   whose names contain an H, such as Arg `NH1`/`NH2` and Tyr `OH`, which are
   precisely the atoms that form interface contacts.
6. **The hard-coded 99-frame score array** is now sized from the data, so
   `--sim_time` and the frame count can change without silently truncating or
   zero-padding the scores.
7. **Velocities are initialised** from the Maxwell-Boltzmann distribution
   *(changes results)* before equilibration and production, instead of starting
   from zero velocities.
8. **Scoring windows are expressed in nanoseconds**, so "first 1 ns" and
   "last 2 ns" stay correct when `--sim_time` changes.
9. `collect_results` no longer assumes contiguous replica numbering, skips
   missing or empty files instead of crashing, and pools replicas with
   `np.concatenate` so unequal lengths cannot raise.
10. Selections are validated up front with actionable errors, instead of
   failing deep inside the analysis.
11. The stale-bias-file hazard on restart is now reported.
12. Written files are closed properly, and the aliasing between
    `input_positions` and `positions` during restraint setup was removed.
13. Docstrings corrected: the integrator uses 2 fs (not 4 fs), equilibration
    is 1 ns (not 500 ps), and bias is deposited every 0.5 ps (not 1 ps).
14. Added GPU platform selection, a per-replica results file, and a
    per-complex error summary so one bad complex does not abort a batch.

---

## Citing

If you use this pipeline, please cite the original BPMD work:

> Clark, A. J.; Tiwary, P.; Borrelli, K.; Feng, S.; Miller, E. B.; Abel, R.;
> Friesner, R. A.; Berne, B. J. *Prediction of Protein–Ligand Binding Poses via
> a Combination of Induced Fit Docking and Metadynamics Simulations.*
> J. Chem. Theory Comput. **2016**, 12, 2990–2998.
> https://doi.org/10.1021/acs.jctc.6b00201

and the OpenMM implementation it is based on:

> Lukauskis, D.; Samways, M. L.; Aureli, S.; Cossins, B. P.; Taylor, R. D.;
> Gervasio, F. L. *Open Binding Pose Metadynamics: An Effective Approach for
> the Ranking of Protein–Ligand Binding Poses.*
> J. Chem. Inf. Model. **2022**, 62, 6209–6216.

together with [OpenMM](https://openmm.org), [AmberTools](https://ambermd.org),
[pdb2pqr/PROPKA](https://pdb2pqr.readthedocs.io),
[MDAnalysis](https://www.mdanalysis.org) and [MDTraj](https://www.mdtraj.org).

## License

MIT — see [LICENSE](LICENSE).
