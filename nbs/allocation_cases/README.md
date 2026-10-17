# Cloud allocation notebooks

Eight gsim notebooks generate real meshes/configs and exercise cloud allocation for FDTD and Palace. Each supported
notebook prepares the job with measured sizing requirements, uploads the solver inputs, and submits the job. Unsupported
notebooks attempt preparation and verify an oversized-sizing rejection. They do not start oversized solves.

| Notebook                                       | Geometry               | Sizing controls                        |
| ---------------------------------------------- | ---------------------- | -------------------------------------- |
| [FDTD small](fdtd_small.ipynb)                 | Straight waveguide     | Approximately 50 million Yee cells     |
| [FDTD medium](fdtd_medium.ipynb)               | Straight waveguide     | Approximately 500 million Yee cells    |
| [FDTD large](fdtd_large.ipynb)                 | Straight waveguide     | Approximately 1 billion Yee cells      |
| [FDTD unsupported](fdtd_unsupported.ipynb)     | Straight waveguide     | Approximately 20 billion Yee cells     |
| [Palace small](palace_small.ipynb)             | IHP coplanar waveguide | Coarse tetrahedral mesh, field order 2 |
| [Palace medium](palace_medium.ipynb)           | IHP coplanar waveguide | Finer mesh, field order 4              |
| [Palace large](palace_large.ipynb)             | IHP coplanar waveguide | Further mesh refinement, field order 4 |
| [Palace unsupported](palace_unsupported.ipynb) | IHP coplanar waveguide | Finest mesh, field order 4             |

These are allocation and launch checks, with short FDTD runs or one Palace frequency, rather than converged accuracy
examples. The larger Palace cases use higher field order to exercise larger Field DOF counts while reducing local
tetrahedral meshing work. Generating those meshes still takes time and RAM. Exact counts depend on Gmsh and PDK
versions.

## Local input measurements

These measurements were generated with gsim source based on `9c044559`, GDSFactory 9.40.2, Gmsh 4.15.0, and IHP PDK
0.2.8. All eight input-generation paths were executed locally. Submission cells were checked with mocked HTTP; no cloud
solves were launched during notebook validation.

| Case               | Requested Yee cells | Tetrahedra | Estimated Field DOFs | Uncompressed inputs, MiB |
| ------------------ | ------------------: | ---------: | -------------------: | -----------------------: |
| FDTD small         |          50,000,000 |          - |                    - |                     0.02 |
| FDTD medium        |         500,500,000 |          - |                    - |                     0.10 |
| FDTD large         |       1,001,000,000 |          - |                    - |                     0.19 |
| FDTD unsupported   |      20,000,000,000 |          - |                    - |                     4.03 |
| Palace small       |                   - |     34,862 |              223,152 |                     1.67 |
| Palace medium      |                   - |    272,308 |           11,137,888 |                    14.39 |
| Palace large       |                   - |  1,240,018 |           50,792,952 |                    68.40 |
| Palace unsupported |                   - |  5,244,924 |          214,502,736 |                   289.81 |

FDTD targets can round up by one cell along an axis because its written geometry bounds and grid spacing are
floating-point values. The request always uses the measured count rather than the nominal target.

## Run

Install the project dependencies, including the `docs` and `test` dependency groups for a notebook kernel and the public
IHP PDK. Use this checkout as the kernel's gsim installation.

Set these variables in the notebook kernel environment:

- `SIMULATION_API_URL`: the simulation API base immediately before `/job/prepare`, ending in `/v1`.
- `GFP_API_KEY`: your existing API key.

Open the notebooks from `nbs/allocation_cases` so the shared `helpers.py` can be imported. Run the cells in order.
Supported cases submit a real job in the final cell; there is no submission toggle. Re-running that cell creates another
job. The returned job ID lets you track completion in the cloud job view.

## Metadata and allocation

FDTD sends only its measured total Yee-cell count, including PML. Palace sends measured tetrahedra and the
topology-based input Field DOF estimate from gsim. Palace's final Field DOFs after preprocessing can differ from that
estimate. Counts are read from the generated inputs, never replaced with targets.

The small/medium/large labels describe expected test outcomes. The cloud service owns the profiles and selects
resources. No allocation thresholds, EC2 instance choices, or pricing logic live in these examples. Inspect the printed
measurements and the requested resources returned by the server, then check the job in the admin view. HTTP success
alone does not prove that the expected profile was used. If profiles or mesh counts change, adjust the example's
geometry/grid/mesh settings and repeat.

The notebooks use a small HTTP adapter because older SDK upload paths omit `sizing_requirements`. The target server must
support this field and have sizing profiles enabled. An unrelated HTTP 400, an authorization failure, or an unexpectedly
accepted oversized case fails the notebook rather than being treated as a successful rejection.

Each notebook writes an ignored `output/<case>/` directory containing:

- `inputs/`: generated mesh/config and, for Palace, gsim's richer mesh diagnostics.
- `metadata.json`: the sizing object sent to the API.
- `prepare.json`: the complete prepare request.
- `source.zip`: only the mesh and solver config, created for supported cloud submissions.
- `job-id.txt` and `cloud-result.json`: job diagnostics, without credentials or presigned URLs.

The metadata sidecar stays outside the ZIP. This adapter sends its JSON directly to the API; it does not upload a
separate metadata object to S3. That upload requires SDK/backend support for a sidecar upload URL. Solver input upload
and submission work through the existing prepare/upload/submit flow.

## Disk use

Allocation size is not a disk estimate. FDTD uploads a tetrahedral geometry mesh rather than its full Yee grid, so its
input ZIP can stay small even when the grid needs a large amount of RAM. Palace input files grow with tetrahedral mesh
size; increasing field order increases Field DOFs without adding geometry elements to the mesh file.

These Palace cases disable full field export (`SaveStep = 0`) and solve one frequency. Actual result sizes and peak
solver scratch space have not been measured by local input generation. Saving fields, adding frequency samples, or
changing solver scratch settings can change disk requirements independently of the allocation category.
