# Electronic-integrated-circuit benchmark data

This directory contains only small reference files whose public source and integrity can be recorded. IHP artifacts are
deliberately excluded: the pinned GitHub repository does not declare a redistribution license, so the IHP notebook
downloads and verifies those files at runtime.

## NIST Parylene-C CPW

Source: NIST dataset [10.18434/mds2-2817](https://doi.org/10.18434/mds2-2817), whose catalog record identifies the
dataset as publicly accessible under the NIST license. The original archive is 191,909,838 bytes with SHA-256
`d57a0131365b6e9790898903ca7dec96782ddac1e8436747ecfd594b89438bdd`.

The five CSV files are the published ANSYS Q2D RLCG tables for the 2.5 um-gap air-channel case.
`nist/air_data_vs_sim.npz` is a lossless reshaping of the four complex columns in `air_data_vs_sim.mat`; its arrays are
named `simulation_frequency_hz`, `simulation_s`, `measurement_frequency_hz`, and `measurement_s`, with each S matrix
stored in conventional row/column order. The source MAT file has SHA-256
`1831bf5d373f6123a313ef5fa7037c00b13bb2aeb0aa4ce7f93ad8a2f8be1165`.

The MAT reference predates the CSV files in the published archive (July 11 versus July 21/October 6, 2022). Re-running
the included cascade using the final CSV tables therefore differs from the stored simulation by at most 0.001765 in
absolute complex S, rather than bit-for-bit.

| File                         | SHA-256                                                            |
| ---------------------------- | ------------------------------------------------------------------ |
| `nist/1_NP_25umgap.csv`      | `16bbaf0d68dcf4f3aee44371f52c723f9233d41cbc4641a588df57c29cb93566` |
| `nist/2_YP_25umgap.csv`      | `d598f44989c3cd06bb5eade14da13dcb259eeab7035d7ffb16cc4c09f8506c9f` |
| `nist/3_AirPDMS_25umgap.csv` | `3535574abae7017956ea9b1343437fb5776a9f2829c7caecdb606989fe80bc74` |
| `nist/4_PDMS_25umgap.csv`    | `6779f8ff1077debc149eb8c97c62ba1fff87ed0b412eeeca878a38e2e1a8eaf4` |
| `nist/5b_AirChannel.csv`     | `400899e0767ee9c2ea35265c1df74fe9653a38f0f3f3402e6b062ccde18bef6f` |
| `nist/air_data_vs_sim.npz`   | `8b345f9ce60993d356ad1e5cbaf337f56e4d290191db2897f4ccd9043c79103a` |
