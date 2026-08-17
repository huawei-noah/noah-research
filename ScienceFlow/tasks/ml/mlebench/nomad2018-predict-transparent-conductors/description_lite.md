# Nomad2018 Predict Transparent Conductors — Lite Task Description

## Task description
Innovative materials design is needed to tackle some of the most important health, environmental, energy, social, and economic challenges of this century. In particular, improving the properties of materials that are intrinsically connected to the generation and utilization of energy is crucial if we are to mitigate …

## Task objective
- **Input:** Each test row / notebook identified by `id` (and any features in released `test` / `public` data).
- **Output:** For each id in the test set, you must predict a value for both formation_energy_ev_natom and bandgap_energy_ev.

## Target metric (evaluation)
Submissions are evaluated on the column-wise root mean squared logarithmic error. The RMSLE for a single column calculated as where: \\(n\\) is the total number of observations \\(p_i\\) is your prediction \\(a_i\\) is the actual value \\(\log(x)\\) is the natural logarithm of …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,formation_energy_ev_natom,bandgap_energy_ev; one row per test key (see `sample_submission.csv`).
```
id,formation_energy_ev_natom,bandgap_energy_ev
1,0.1779,1.8892
2,0.1779,1.8892
```

## Dataset and construction
- Spacegroup (a label identifying the symmetry of the material)
- Total number of Al, Ga, In and O atoms in the unit cell ( )
- Relative compositions of Al, Ga, and In (x, y, z)
- Lattice vectors and angles: lv1, lv2, lv3 (which are lengths given in units of angstroms (  meters) and   (which are angles in degrees between 0° and 360°)
