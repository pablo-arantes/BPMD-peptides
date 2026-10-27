#!/usr/bin/env python3
"""Generate BPMD_peptides_Colab.ipynb.

The notebook is generated rather than hand-edited so that the cell text stays
reviewable in a diff: editing .ipynb JSON by hand is how stray outputs,
execution counts and broken source arrays creep in.

    python tools/make_colab_notebook.py      # run from the repository root

After regenerating, check it still loads:

    python -c "import json,nbformat; nbformat.validate(nbformat.reads(
        json.dumps(json.load(open('BPMD_peptides_Colab.ipynb'))), as_version=4))"
"""
import json

def as_source(src):
    """nbformat source lines keep their trailing newline; the last one does not.

    Without the newlines every cell collapses onto a single line, and since
    each code cell opens with a '#@title' comment the entire cell would be
    swallowed by that comment and silently do nothing.
    """
    lines = src.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": as_source(src)}

def code(src, title=None):
    meta = {}
    if title:
        meta["cellView"] = "form"
    return {"cell_type": "code", "execution_count": None, "metadata": meta,
            "outputs": [], "source": as_source(src)}

cells = []

cells.append(md(r"""
# OpenBPMD for peptides — run it on Google Colab

Is your docked peptide pose actually **stable**? This notebook answers that with
binding pose metadynamics: a small bias is applied to the peptide's RMSD, which
actively tries to push the peptide out of its pose. A well-bound peptide
resists; a weakly docked one slides away and loses its contacts.

You need no GPU, no cluster and no local install — just a structure file.

**Repository:** https://github.com/pablo-arantes/BPMD-peptides

---

### What you get

| Score | Meaning |
|---|---|
| **PoseScore** | RMSD (Å) of the peptide from its starting pose. Lower = pose retained |
| **ContactScore** | Fraction of the original peptide–receptor contacts that survive. Higher = interface intact |
| **CompScore** | `PoseScore − 5 × ContactScore`, a single number for ranking. Lower = more stable |

### Before you start

1. **Turn on the GPU:** `Runtime` → `Change runtime type` → `T4 GPU`. Without
   it this is roughly 20× slower.
2. **Read the two warnings in step 5** about Colab session limits before you
   choose how long to run. Colab disconnects after a few hours, so the defaults
   here are smaller than the published protocol.
3. **Mount Google Drive in step 2** if you want your results to survive a
   disconnect. Replicas that already finished are skipped when you re-run, so a
   long job can be completed across several sessions.

Run the cells in order.
"""))

cells.append(md("## 1. Check the GPU"))

cells.append(code(r"""
#@title Check which GPU Colab gave you { display-mode: "form" }
import subprocess

gpu = subprocess.run(["nvidia-smi",
                      "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"],
                     capture_output=True, text=True)
HAS_GPU = gpu.returncode == 0

if HAS_GPU:
    first = gpu.stdout.strip().split("\n")[0]       # one line per GPU
    name, mem, driver = [x.strip() for x in first.split(",")]
    print(f"GPU            : {name}")
    print(f"Memory         : {mem}")
    print(f"Driver         : {driver}")
    print("\nGood to go. The installer will fetch a CUDA-enabled OpenMM.")
else:
    print("No GPU found in this runtime.")
    print("\nFix it now:  Runtime -> Change runtime type -> T4 GPU -> Save")
    print("then re-run this cell. You CAN continue on CPU, but a 10 ns")
    print("replica takes many hours instead of minutes.")
"""))

cells.append(md(r"""
## 2. Install everything (~8–12 minutes, once per session)

This installs AmberTools, OpenMM, PyMOL, pdb2pqr and the analysis libraries
into a self-contained conda environment, then clones the repository.

The environment goes in its own directory and **Colab's own Python is left
alone**, so there is no kernel restart and no broken `numpy`. The two pipeline
scripts are run as subprocesses inside that environment.

This install resolves to a newer stack than the one the repository was
developed against (Python 3.11, OpenMM 8.6, NumPy 2 here, versus Python 3.10,
OpenMM 7.7, NumPy 1.26 locally). The package set is verified to solve, and the
scripts avoid the APIs NumPy 2 removed, but if something looks off this
difference is worth remembering.
"""))

cells.append(code(r"""
#@title Install AmberTools + OpenMM + the pipeline { display-mode: "form" }
import os, subprocess, sys, time

ENV  = "/content/bpmd-env"
REPO = "/content/BPMD-peptides"
MM   = "/content/bin/micromamba"

def sh(cmd, **kw):
    # Run a shell command; raise with its output if it fails.
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise RuntimeError(f"failed: {cmd}")
    return r.stdout

try:
    HAS_GPU
except NameError:
    raise RuntimeError("Run step 1 ('Check which GPU Colab gave you') first -- "
                       "this cell needs to know whether to install the CUDA "
                       "build of OpenMM.")

t0 = time.time()

# --- the pipeline itself ---
if not os.path.isdir(REPO):
    print("Cloning BPMD-peptides ...")
    sh(f"git clone --depth 1 https://github.com/pablo-arantes/BPMD-peptides.git {REPO}")
else:
    # Hard reset rather than pull: the clone is shallow and nobody should be
    # editing it, and a silently failed pull would leave stale code running.
    sh(f"cd {REPO} && git fetch --depth 1 origin main "
       f"&& git reset --hard origin/main")
print("Pipeline version:", sh(f"cd {REPO} && git log --oneline -1").strip())
print("  (if this is not the newest commit, delete /content/BPMD-peptides "
      "and re-run)")

# --- micromamba (the archive holds bin/micromamba, so this lands at MM) ---
if not os.path.isfile(MM):
    print("Fetching micromamba ...")
    sh("curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest "
       "| tar -xj -C /content bin/micromamba")

# --- the scientific stack ---
PKGS = ["python=3.11", "ambertools", "openmm", "pymol-open-source",
        "mdtraj", "mdanalysis", "biopandas", "pdb2pqr"]
if HAS_GPU:
    # Colab's driver is CUDA 12.x; without this pin the solver may pick a
    # CUDA 13 build of OpenMM, which that driver cannot load.
    PKGS.append("cuda-version=12")

if not os.path.isfile(f"{ENV}/bin/tleap"):
    print(f"Creating the environment ({'GPU' if HAS_GPU else 'CPU'} build). "
          f"This is the slow part -- go and get a coffee ...")
    sh(f"{MM} create -y -p {ENV} -c conda-forge --override-channels "
       + " ".join(f'"{p}"' for p in PKGS))
else:
    print("Environment already present, skipping.")

# --- make the executables findable and verify ---
os.environ["PATH"] = f"{ENV}/bin:" + os.environ["PATH"]
PY = f"{ENV}/bin/python"

print(f"\nInstalled in {time.time()-t0:.0f} s. Verifying:")
for exe in ("tleap", "pdb2pqr30"):
    found = sh(f"which {exe} || echo MISSING").strip()
    print(f"  {exe:12s} {found}")
print(subprocess.run([PY, "-c",
      "import openmm, mdtraj, MDAnalysis, biopandas, pymol;"
      "from openmm import Platform;"
      "print('  openmm      ', openmm.__version__);"
      "print('  mdtraj      ', mdtraj.version.short_version);"
      "print('  MDAnalysis  ', MDAnalysis.__version__);"
      "print('  platforms   ', [Platform.getPlatform(i).getName()"
      "                         for i in range(Platform.getNumPlatforms())])"],
      capture_output=True, text=True).stdout, end="")
"""))

cells.append(md(r"""
## 3. Where should the results go?

The simulation runs on Colab's **local disk**, which is fast and reliable, and
the results are **copied up to Google Drive** after the topology is built and
after every replica finishes.

It is tempting to point the whole job at Drive and skip the copying, but that
does not work: Drive is a FUSE network mount, and under sustained writes it
drops out with `Transport endpoint is not connected`, killing the run. It is
also slow — a trajectory written frame by frame over a network mount is
painful. So Drive is used as the place results are *saved to*, not the place
they are *computed in*.

**Mount Drive anyway.** Local disk is wiped when the session ends, and a full
job is longer than one Colab session. With Drive, each finished replica is
banked immediately, and re-running step 7 in a new session carries on from
there. The cell below also checks the mount and remounts it if it has died.
"""))

cells.append(code(r"""
#@title Set up local scratch + Google Drive { display-mode: "form" }
use_google_drive = True  #@param {type:"boolean"}
drive_subfolder = "BPMD_peptides"  #@param {type:"string"}
copy_trajectories_to_drive = True  #@param {type:"boolean"}

import os, shutil, subprocess

# Everything the simulation touches lives on local disk. Drive is only ever
# written to in one go, by save_to_drive() below.
LOCAL = "/content/work"
WORK = LOCAL
os.makedirs(LOCAL, exist_ok=True)

DRIVE = None
if use_google_drive:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = f"/content/drive/MyDrive/{drive_subfolder}"
    os.makedirs(DRIVE, exist_ok=True)


def drive_ok():
    # The mount dies silently during long runs; this is how we notice.
    if DRIVE is None:
        return False
    try:
        os.listdir(DRIVE)
        return True
    except OSError:
        return False


def remount_drive():
    print("  Drive mount is not responding -- remounting ...")
    from google.colab import drive
    try:
        drive.flush_and_unmount()
    except Exception:
        pass
    drive.mount("/content/drive", force_remount=True)
    os.makedirs(DRIVE, exist_ok=True)


def _mirror(src, dst):
    os.makedirs(dst, exist_ok=True)
    skip_dcd = not copy_trajectories_to_drive
    if shutil.which("rsync"):
        cmd = ["rsync", "-a"]
        if skip_dcd:
            cmd += ["--exclude", "*.dcd"]
        r = subprocess.run(cmd + [src + "/", dst + "/"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
        print("  rsync failed, copying file by file:", r.stderr[-200:])
    for root, _, names in os.walk(src):
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target, exist_ok=True)
        for n in names:
            if skip_dcd and n.endswith(".dcd"):
                continue
            shutil.copy2(os.path.join(root, n), os.path.join(target, n))


def save_to_drive(name=None):
    # Bank local results on Drive. Safe to call as often as you like.
    if DRIVE is None:
        return
    if not drive_ok():
        remount_drive()
    src = LOCAL if name is None else os.path.join(LOCAL, name)
    dst = DRIVE if name is None else os.path.join(DRIVE, name)
    if not os.path.isdir(src):
        return
    try:
        _mirror(src, dst)
        print(f"  saved to Drive: {dst}")
    except OSError as e:
        print(f"  WARNING: could not save to Drive ({e}).")
        print(f"  Your results are still in {src}. Re-run this cell to retry,")
        print(f"  but do not end the session before they are saved.")


def restore_from_drive(name):
    # Pull an earlier session's results for one complex back to local disk.
    if DRIVE is None:
        return False
    if not drive_ok():
        remount_drive()
    src = os.path.join(DRIVE, name)
    if not os.path.isdir(src):
        return False
    _mirror(src, os.path.join(LOCAL, name))
    return True


print("Computing in:", LOCAL)
print("Saving to   :", DRIVE or "NOWHERE -- results are lost when this "
                                "session ends")
if DRIVE:
    print("Already on Drive:", sorted(os.listdir(DRIVE)) or "(empty)")
"""))

cells.append(md(r"""
## 4. Your structure

One PDB file containing **the peptide and the receptor together**, in the pose
you want to test. Either fetch it from the PDB by ID, or upload your own
(a docked complex, a model, a frame from another simulation).

Requirements, and the reasons they matter:

- **The peptide and receptor must be separate chains** — separated by a `TER`
  record or by different chain IDs. This is how the script tells the two
  molecules apart.
- **Keep the header `SSBOND` records** if the structure has disulfides. They
  are read from the header, which pdb2pqr discards later.
- **Model in missing loops first.** `tleap` bonds consecutive residues of a
  chain whatever the distance, so a gap becomes a stretched bond. Step 5 warns
  if it finds one.
- Waters, hydrogens and `OXT` are removed for you.
"""))

cells.append(code(r"""
#@title Fetch from the PDB, or upload your own { display-mode: "form" }
source = "PDB ID"  #@param ["PDB ID", "Upload a file"]
pdb_id = "3i40"  #@param {type:"string"}

import os, shutil, subprocess

INPUT_DIR = os.path.join(WORK, "inputs")
os.makedirs(INPUT_DIR, exist_ok=True)

if source == "PDB ID":
    pdb_id = pdb_id.strip().lower()
    dest = os.path.join(INPUT_DIR, f"{pdb_id}.pdb")
    r = subprocess.run(["curl", "-sSLf", "-o", dest,
                        f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"])
    if r.returncode != 0:
        raise RuntimeError(f"Could not download '{pdb_id}' from the PDB.")
    COMPLEX = pdb_id
else:
    from google.colab import files
    up = files.upload()
    if not up:
        raise RuntimeError("No file uploaded.")
    fname = list(up)[0]
    if not fname.lower().endswith(".pdb"):
        raise RuntimeError("Please upload a .pdb file.")
    COMPLEX = os.path.splitext(os.path.basename(fname))[0]
    dest = os.path.join(INPUT_DIR, f"{COMPLEX}.pdb")
    shutil.move(fname, dest)

PDB_PATH = dest
print(f"Structure: {PDB_PATH}")

save_to_drive("inputs")        # keep a copy of exactly what was run

# If an earlier session already did some of this complex, bring it back
if restore_from_drive(COMPLEX):
    done = os.path.join(LOCAL, COMPLEX, "peptide_pose")
    reps = len([d for d in os.listdir(done)
                if d.startswith("rep_") and os.path.isfile(
                    os.path.join(done, d, "bpm_results.csv"))]) \
        if os.path.isdir(done) else 0
    print(f"\nRestored '{COMPLEX}' from Drive: {reps} finished replica(s). "
          f"Step 7 will carry on from there.")

# Report what is in the file, so surprises surface now and not later
chains, ssbond, hetero = {}, 0, set()
with open(PDB_PATH) as fh:
    for line in fh:
        if line.startswith("SSBOND"):
            ssbond += 1
        elif line.startswith("ATOM"):
            chains.setdefault(line[21], set()).add(line[22:27])
        elif line.startswith("HETATM"):
            hetero.add(line[17:20].strip())
print(f"\nChains with protein atoms ({len(chains)}):")
for ch, residues in chains.items():
    print(f"  chain '{ch}': {len(residues)} residues")
print(f"SSBOND records: {ssbond}")
print(f"HETATM groups : {sorted(hetero) or '(none)'}")
if len(chains) < 2:
    print("\nWARNING: fewer than two protein chains found. This pipeline needs "
          "the peptide and the receptor as separate chains.")
if hetero - {"HOH"}:
    print("\nNOTE: non-water HETATM groups are kept and must be "
          "parameterisable by tleap (gaff2/GLYCAM). Remove ligands and "
          "cofactors you do not want, or tleap will stop with an error.")
"""))

cells.append(md(r"""
## 5. Stage 1 — build the AMBER topology

pdb2pqr/PROPKA assign protonation states at your pH, then `tleap` adds the
force field, solvates in a truncated octahedron, neutralises the system and
adds salt.

`--ssbond_complex` matters for peptides: leave it **no** for a normal,
non-covalent peptide ligand. Set it to **yes** only for a genuinely covalent
complex — a disulfide-tethered peptide cannot leave its pose, which makes the
scores meaningless.
"""))

cells.append(code(r"""
#@title Topology settings, then build { display-mode: "form" }
force_field = "ff14SB"  #@param ["ff14SB", "ff19SB"]
ph = 7.4  #@param {type:"number"}
titration_method = "propka"  #@param ["propka", "none"]
inter_chain_disulfides = "no"  #@param ["no", "yes"]
box_buffer_angstrom = 10.0  #@param {type:"number"}
salt_concentration_M = 0.15  #@param {type:"number"}
rebuild_if_already_built = False  #@param {type:"boolean"}

import subprocess, sys

PRMTOP = f"{WORK}/{COMPLEX}/{COMPLEX}_amber.prmtop"
INPCRD = f"{WORK}/{COMPLEX}/{COMPLEX}_amber.inpcrd"

def already_built():
    # Both files, and not truncated: a dropped Drive mount can leave a
    # half-written topology behind, which must not be mistaken for a good one.
    return (os.path.isfile(PRMTOP) and os.path.getsize(PRMTOP) > 1000
            and os.path.isfile(INPCRD) and os.path.getsize(INPCRD) > 100)

if already_built() and not rebuild_if_already_built:
    print(f"Topology already built: {PRMTOP}")
    print("Skipping. (tleap places the salt ions randomly, so rebuilding "
          "makes a topology that does not match replicas you have already "
          "run. Tick 'rebuild_if_already_built' only if you mean it.)")
    raise SystemExit  # ends this cell only, not the notebook

cmd = [PY, "-u", f"{REPO}/build_topology.py",   # -u: stream progress live
       "--input_dir", INPUT_DIR, "--work_dir", WORK,
       "--pdb", PDB_PATH,
       "--ff", force_field,
       "--ph", str(ph),
       "--titration_method", titration_method,
       "--ssbond_complex", inter_chain_disulfides,
       "--size_box", str(box_buffer_angstrom),
       "--concentration", str(salt_concentration_M)]
print(" ".join(cmd), "\n")

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env=os.environ)
for line in proc.stdout:          # stream it, the run takes a minute or two
    sys.stdout.write(line)
proc.wait()

if proc.returncode != 0 or not os.path.isfile(PRMTOP):
    raise RuntimeError(
        "Topology generation failed. Scroll up for the reason; the tleap logs "
        f"are in {WORK}/{COMPLEX}/*.log. A 'Transport endpoint is not "
        "connected' error means the Drive mount dropped -- re-run step 3 to "
        "remount, then this cell again.")

print(f"\nTopology ready: {PRMTOP}")
save_to_drive(COMPLEX)
"""))

cells.append(md(r"""
## 6. Which chain is the peptide?

The cell below lists the chains as the simulation will see them, numbered in
file order. **Read the list, then set `peptide_chain` in step 7.**

This is deliberately a manual step. Picking the wrong molecule does not
produce an error — it produces perfectly plausible scores for the wrong thing.
"""))

cells.append(code(r"""
#@title List the chains found in the prepared structure { display-mode: "form" }
import subprocess

ED_PDB = f"{WORK}/{COMPLEX}/{COMPLEX}_ed.pdb"
out = subprocess.run(
    [PY, "-c",
     "import sys; sys.path.insert(0, sys.argv[1]);"
     "from run_bpmd import chain_ranges_from_pdb;"
     "rs = chain_ranges_from_pdb(sys.argv[2]);"
     "print('\\n'.join(f'{i}\\t{s}\\t{e}' for i,(s,e) in enumerate(rs, 1)))",
     REPO, ED_PDB],
    capture_output=True, text=True, env=os.environ)
if out.returncode != 0:
    print(out.stderr); raise RuntimeError("Could not inspect the chains.")

rows = [l.split("\t") for l in out.stdout.strip().split("\n") if l.strip()]
print(f"Chains in {COMPLEX}_ed.pdb, numbered as the simulation sees them:\n")
print(f"  {'chain':>5}  {'residues':>12}  {'count':>6}")
for idx, start, end in rows:
    print(f"  {idx:>5}  {start:>5} - {end:<5}  {int(end)-int(start)+1:>6}")
print(f"\nUse these numbers for 'peptide_chain' below"
      f"{' (receptor is inferred automatically for two chains)' if len(rows)==2 else ''}.")
if len(rows) > 2:
    print("More than two chains: you must set 'receptor_chain' as well.")
"""))

cells.append(md(r"""
## 7. Stage 2 — run the metadynamics

### Choose your settings with Colab's limits in mind

The published protocol is **10 replicas × 10 ns**. On a T4 that is well beyond
one Colab session for a typical complex, so the defaults below are smaller.

- **Nothing is wasted.** Replicas are run one at a time and each is copied
  to Drive the moment it is scored. If Colab disconnects, re-run this cell in
  a new session and it resumes at the next unfinished replica.
- **Start small.** Run `replicas = 1`, `sim_time = 1` first to confirm the
  whole thing works on your system and to measure the speed, then increase.
- **The scoring window is the last 2 ns**, so `sim_time` below 2 ns averages
  whatever is there and is only useful as a smoke test.
- **Use the same `sim_time` when you resume**, otherwise the window is
  computed from the wrong trajectory length.
- **`timestep_fs` is 4 fs**, which is what hydrogen mass repartitioning is
  for: it halves the wall time without changing the metadynamics protocol,
  which is defined in picoseconds. Drop it to 2 fs only to reproduce runs
  made before this was fixed. Equilibration always uses 2 fs.
""")) 

cells.append(code(r"""
#@title Simulation settings, then run { display-mode: "form" }
peptide_chain = 1  #@param {type:"integer"}
receptor_chain = 0  #@param {type:"integer"}
replicas = 2  #@param {type:"integer"}
sim_time_ns = 2.0  #@param {type:"number"}
equilibration_ps = 500.0  #@param {type:"number"}
hill_height_kcal = 0.5  #@param {type:"number"}
temperature_K = 300.0  #@param {type:"number"}
timestep_fs = 4.0  #@param {type:"number"}

import glob, subprocess, sys, time

POSE_DIR = f"{WORK}/{COMPLEX}/peptide_pose"

base_cmd = [PY, "-u", f"{REPO}/run_bpmd.py",    # -u: stream progress live
            "--work_dir", WORK, "--complex", COMPLEX,
            "--peptide_chain", str(peptide_chain),
            "--sim_time", str(sim_time_ns),
            "--equil_time", str(equilibration_ps),
            "--hill_height", str(hill_height_kcal),
            "--temperature", str(temperature_K),
            "--timestep", str(timestep_fs),
            "--platform", "CUDA" if HAS_GPU else "CPU"]
if receptor_chain > 0:                 # 0 means "infer it"
    base_cmd += ["--receptor_chain", str(receptor_chain)]


def finished_replicas():
    return sorted(glob.glob(os.path.join(POSE_DIR, "rep_*",
                                         "bpm_results.csv")))


# One replica per call, so each one is banked on Drive as soon as it is
# scored. The script skips replicas that are already done, which is what
# makes this resumable across sessions. '--nreps n' means "make sure n
# replicas exist", so asking for one more than are finished adds exactly one.
t0 = time.time()
for _ in range(replicas):
    before = len(finished_replicas())
    if before >= replicas:
        break
    print(f"\n{'='*64}")
    print(f"  replica {before + 1} of {replicas}   "
          f"(elapsed {(time.time()-t0)/60:.1f} min)")
    print(f"{'='*64}")

    proc = subprocess.Popen(base_cmd + ["--nreps", str(before + 1)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=os.environ)
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.wait()

    save_to_drive(COMPLEX)        # bank whatever progress was made

    if proc.returncode != 0:
        raise RuntimeError(
            "The run failed -- scroll up for the reason. Anything already "
            "finished has been saved to Drive, so re-running this cell "
            "resumes rather than starting over. A 'Transport endpoint is "
            "not connected' error means the Drive mount dropped: re-run "
            "step 3 to remount, then this cell.")
    if len(finished_replicas()) == before:
        raise RuntimeError(
            f"Replica {before + 1} did not produce a bpm_results.csv and the "
            f"script reported no error. Stopping rather than looping.")

print(f"\n{len(finished_replicas())} replica(s) finished in "
      f"{(time.time()-t0)/60:.1f} min")
print("Results in:", POSE_DIR)
"""))

cells.append(md("## 8. Results"))

cells.append(code(r"""
#@title Final scores { display-mode: "form" }
import pandas as pd, os

POSE_DIR = f"{WORK}/{COMPLEX}/peptide_pose"
final = pd.read_csv(f"{POSE_DIR}/results.csv")
per_rep = pd.read_csv(f"{POSE_DIR}/results_per_replica.csv")

print(f"=== {COMPLEX}: averaged over the last 2 ns of "
      f"{int(final['nreps'][0])} replica(s) ===")
display(final)
print("=== per replica (check these agree with each other) ===")
display(per_rep)

pose, contact = final["PoseScore"][0], final["ContactScore"][0]
spread = per_rep["CompScore"].max() - per_rep["CompScore"].min()

print("Reading of these numbers:")
if pose <= 2 and contact >= 0.7:
    print(f"  PoseScore {pose:.2f} A and ContactScore {contact:.2f}: the peptide")
    print("  held its pose against the bias -- consistent with a stable pose.")
elif contact >= 0.7:
    print(f"  PoseScore {pose:.2f} A but ContactScore {contact:.2f}: the peptide")
    print("  rearranged yet stayed on the interface. Often the right site with")
    print("  the wrong modelled pose -- watch the trajectory.")
else:
    print(f"  PoseScore {pose:.2f} A with ContactScore {contact:.2f}: the bias")
    print("  pushed the peptide out and the interface did not recover.")
if len(per_rep) < 3:
    print(f"\n  Only {len(per_rep)} replica(s): too few to judge reproducibility.")
elif spread > 2:
    print(f"\n  CompScore spread across replicas is {spread:.2f} -- not")
    print("  converged. Run more replicas before ranking anything on this.")
"""))

cells.append(code(r"""
#@title Plot the scores over time { display-mode: "form" }
import glob, os, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

SCORE_WINDOW_NS = 2.0
POSE_DIR = f"{WORK}/{COMPLEX}/peptide_pose"

def replica_number(path):
    m = re.search(r"rep_(\d+)$", os.path.basename(os.path.dirname(path)))
    return int(m.group(1)) if m else 0

files = sorted(glob.glob(os.path.join(POSE_DIR, "rep_*", "bpm_results.csv")),
               key=replica_number)
reps = {f"rep {replica_number(f)}": pd.read_csv(f) for f in files}
if not reps:
    raise RuntimeError(f"No finished replicas in {POSE_DIR}")

n = max(len(df) for df in reps.values())
t = np.arange(1, n + 1) * (sim_time_ns / n)          # frames are evenly spaced
window_start = max(0.0, sim_time_ns - SCORE_WINDOW_NS)

panels = [("PoseScore", "PoseScore  (Å)", 2.0, "2 Å guide"),
          ("ContactScore", "ContactScore", 0.7, "0.7 guide"),
          ("CompScore", "CompScore", None, None)]
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

for ax, (col, ylabel, guide, guide_label) in zip(axes, panels):
    stacked = []
    for label, df in reps.items():
        y = df[col].values
        ax.plot(t[:len(y)], y, lw=1, alpha=0.5, label=label)
        padded = np.full(n, np.nan); padded[:len(y)] = y
        stacked.append(padded)
    ax.plot(t, np.nanmean(np.vstack(stacked), axis=0), lw=2.5, color="k",
            label="mean")
    ax.axvspan(window_start, sim_time_ns, color="tab:orange", alpha=0.12,
               label=f"scored window (last {SCORE_WINDOW_NS} ns)")
    if guide is not None:
        ax.axhline(guide, ls="--", lw=1, color="tab:red", label=guide_label)
    ax.set_ylabel(ylabel); ax.grid(alpha=0.25)

axes[0].legend(loc="upper left", fontsize=7, ncol=2)
axes[0].set_title(f"{COMPLEX} — pose stability over {len(reps)} replica(s)")
axes[-1].set_xlabel("time (ns)")
fig.tight_layout()
out_png = os.path.join(POSE_DIR, "scores.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("Saved:", out_png)
save_to_drive(COMPLEX)          # bank the figure alongside the scores
plt.show()
"""))

cells.append(md(r"""
## 9. Look at the trajectory

Do this before you believe any number. The PoseScore cannot tell a peptide
that unbinds from one that slides along a groove — but you can see the
difference in two seconds.
"""))

cells.append(code(r"""
#@title Download the results { display-mode: "form" }
what_to_download = "scores and plots only"  #@param ["scores and plots only", "everything including trajectories"]

import os, shutil
from google.colab import files

base = f"{WORK}/{COMPLEX}"
stem = f"/content/{COMPLEX}_bpmd_results"

if what_to_download == "everything including trajectories":
    print("Zipping everything (trajectories can be hundreds of MB) ...")
    shutil.make_archive(stem, "zip", base)
else:
    staging = "/content/_bpmd_small"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(f"{staging}/{COMPLEX}/peptide_pose", exist_ok=True)
    for pattern in ("results.csv", "results_per_replica.csv", "scores.png"):
        src = f"{base}/peptide_pose/{pattern}"
        if os.path.isfile(src):
            shutil.copy(src, f"{staging}/{COMPLEX}/peptide_pose/")
    for extra in ("output_info.txt", f"{COMPLEX}_ed.pdb"):
        if os.path.isfile(f"{base}/{extra}"):
            shutil.copy(f"{base}/{extra}", f"{staging}/{COMPLEX}/")
    for rep in sorted(os.listdir(f"{base}/peptide_pose")):
        src = f"{base}/peptide_pose/{rep}/bpm_results.csv"
        if os.path.isfile(src):
            os.makedirs(f"{staging}/{COMPLEX}/peptide_pose/{rep}", exist_ok=True)
            shutil.copy(src, f"{staging}/{COMPLEX}/peptide_pose/{rep}/")
    shutil.make_archive(stem, "zip", staging)

size_mb = os.path.getsize(f"{stem}.zip") / 1e6
print(f"{stem}.zip  ({size_mb:.1f} MB)")
files.download(f"{stem}.zip")
"""))

cells.append(md(r"""
To inspect the pose yourself, load these two files into PyMOL, ChimeraX or VMD
locally — they are in `<complex>/peptide_pose/` on your Drive:

- `centred_equil_system.pdb` — the topology
- `rep_1/trj.dcd` — the trajectory

```
# in PyMOL, locally
load centred_equil_system.pdb
load_traj rep_1/trj.dcd
```

---

## Running the full protocol

The published protocol is 10 replicas × 10 ns. To get there on Colab:

1. Mount Drive (step 3) so each replica is banked as it finishes.
2. Set `replicas = 10` and `sim_time_ns = 10`, and keep those values fixed —
   the scoring window is derived from `sim_time_ns`, so changing it between
   sessions averages the wrong part of the trajectory.
3. Re-run steps 3, 4 and 7 whenever the session drops. Step 4 pulls your
   finished replicas back from Drive and step 7 carries on from there.

If a replica is interrupted mid-run, **delete its whole `rep_N` directory**
before resuming. A partial replica leaves metadynamics bias files behind, and
OpenMM reads them back in, which would continue the old biased run instead of
starting a clean one. The script warns you when it sees this.

## Citing

Please cite the original binding pose metadynamics work:

> Clark, A. J.; Tiwary, P.; Borrelli, K.; Feng, S.; Miller, E. B.; Abel, R.;
> Friesner, R. A.; Berne, B. J. *Prediction of Protein–Ligand Binding Poses via
> a Combination of Induced Fit Docking and Metadynamics Simulations.*
> J. Chem. Theory Comput. **2016**, 12, 2990–2998.
> https://doi.org/10.1021/acs.jctc.6b00201

and the OpenMM implementation this is based on:

> Lukauskis, D.; Samways, M. L.; Aureli, S.; Cossins, B. P.; Taylor, R. D.;
> Gervasio, F. L. *Open Binding Pose Metadynamics: An Effective Approach for
> the Ranking of Protein–Ligand Binding Poses.*
> J. Chem. Inf. Model. **2022**, 62, 6209–6216.

together with [OpenMM](https://openmm.org), [AmberTools](https://ambermd.org),
[pdb2pqr/PROPKA](https://pdb2pqr.readthedocs.io),
[MDAnalysis](https://www.mdanalysis.org) and [MDTraj](https://www.mdtraj.org).
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True,
                  "name": "BPMD_peptides.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

with open("BPMD_peptides_Colab.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(f"wrote BPMD_peptides_Colab.ipynb with {len(cells)} cells "
      f"({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
