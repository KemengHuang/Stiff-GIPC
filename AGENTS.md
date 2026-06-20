# StiffGIPC — Agent Guide

> This file is written for AI coding agents that need to work on the StiffGIPC repository. It summarizes the project structure, build process, runtime behavior, coding conventions, and things to watch out for when making changes.

---

## Project Overview

StiffGIPC is a **C++/CUDA** research implementation of the paper *"StiffGIPC: Advancing GPU IPC for Stiff Affine-Deformable Simulation"* (ACM Transactions on Graphics, 2025). It implements a unified, GPU-accelerated Incremental Potential Contact (IPC) framework that simulates soft bodies, stiff/rigid bodies, cloth, and hybrid couplings using affine body dynamics (ABD) combined with finite element method (FEM) deformable objects.

### What the program does

The compiled executable (`gipc`) loads a scene description and tetrahedral/triangle meshes, then runs a physics time-stepping loop on the GPU: collision detection (broad-phase via LBVH and narrow-phase CCD), barrier contact/ground collision, friction, Newton-type implicit integration, and a PCG solver. It renders the current state interactively with OpenGL/FreeGLUT and can write out surface meshes/screenshots for offline rendering.

### License note

The code is released under **MPLv2.0**. The README explicitly states that commercial use requires contacting the authors.

---

## Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Languages | C++17, CUDA C++17 | Heavy use of CUDA kernels and Thrust/CUB primitives |
| Build system | CMake ≥ 3.18 | Single top-level `CMakeLists.txt` |
| GPU compute | NVIDIA CUDA ≥ 11.0 | Uses cuSPARSE, cuBLAS, cuSOLVER |
| Math/linear algebra | Eigen 3.4.0 | Host-side dense math; also used inside CUDA via muda |
| Visualization | FreeGLUT 3.4.0 + GLEW 2.2.0 + OpenGL | Interactive viewer in `gl_main.cu` |
| JSON | nlohmann/json | Scene and config files |
| Partitioning | METIS + GKlib | Preprocess meshes for domain decomposition (in `MeshProcess/`) |
| CUDA wrapper library | muda | Header-only CUDA utility layer vendored under `StiffGIPC/muda/` |

### Build configurations hard-coded in CMake

The top-level `CMakeLists.txt` always defines these preprocessor switches for the `gipc` target:

- `USE_SNK1`
- `ADAPTIVE_KAPPA`
- `USE_FRICTION`
- `USE_QUADRATIC_BENDING`

If you need to toggle physics features, you must edit `CMakeLists.txt`.

---

## Repository Layout

```
Stiff-GIPC/
├── CMakeLists.txt              # Top-level build configuration
├── README.md                   # Human-facing project description & citation info
├── Assets/                     # Input meshes, sorted mesh partitions, scene files, configs
│   ├── scene/                  # parameterSetting.txt, JSON scenes, abd_system_config.json
│   ├── sorted_mesh/            # METIS-partitioned surface meshes (.obj/.msh + .part)
│   ├── tetMesh/                # Tetrahedral input meshes (.msh)
│   └── triMesh/                # Triangle/cloth input meshes (.obj)
├── MeshProcess/                # Mesh preprocessing tools
│   ├── CMakeLists.txt
│   ├── External/               # Vendored METIS + GKlib
│   └── metis_partition/        # CLI tool that partitions meshes with METIS
└── StiffGIPC/                  # Main simulation source code
    ├── gl_main.cu              # Entry point: GLUT window, rendering, user input
    ├── GIPC.cu / GIPC.cuh      # Core IPC data structures and per-step logic
    ├── ACCD.cu / ACCD.cuh      # Adaptive Continuous Collision Detection
    ├── FrictionUtils.cuh       # Friction-related CUDA helpers
    ├── mlbvh.cu / mlbvh.cuh    # Linear BVH broad-phase collision detection
    ├── femEnergy.cu/.cuh       # FEM elasticity energy/gradient/Hessian
    ├── device_fem_data.cu/.cuh # Device FEM mesh/state representation
    ├── load_mesh.cpp/.h        # Host-side mesh I/O (OBJ/MSH)
    ├── generateVideo.py        # (Unused in build) utility script
    ├── abd_system/             # Affine body dynamics system
    ├── cuda_tools/             # Small CUDA helper wrappers
    ├── gipc/                   # Common types, statistics, timer, scene importer, utilities
    ├── linear_system/          # Global linear system assembly + PCG solver + preconditioners
    └── muda/                   # Vendored muda CUDA framework (header-only)
```

### Module responsibilities

- **`gl_main.cu`** — Creates the OpenGL window, loads meshes via `SimpleSceneImporter`, initializes CUDA, builds the IPC system, and runs the per-frame simulation loop. It is also the only file with a `main()`.
- **`GIPC.cu/.cuh`** — The main `GIPC` class holding device pointers for vertices, faces, edges, collision pairs, barrier/friction data, and the global triplet Hessian. Implements barrier energy, gradients, Hessians, line search, CCD, and Newton step logic.
- **`abd_system/`** — `ABDSystem` and `ABDSimData`: affine-body state (`q`, `q_tilde`, `q_prev`, `q_v`), Jacobians, dyadic mass, gravity, shape/kinetic energy, and system assembly. Contains the math that maps between affine DOFs and world positions.
- **`linear_system/`** — `GlobalLinearSystem` assembles FEM + ABD subsystems (`fem_linear_subsystem`, `abd_linear_subsystem`) into a global matrix and solves it with a PCG solver (`pcg_solver`) and preconditioners (diagonal, ABD block-diagonal, FEM mass).
- **`mlbvh.cu/.cuh`** — Broad-phase collision detection using a linear BVH on the GPU.
- **`ACCD.cu/.cuh`** — Narrow-phase continuous collision detection and largest-feasible-step-size queries.
- **`MeshProcess/`** — Offline preprocessing: takes a tetrahedral mesh and writes a METIS partition file used by the FEM preconditioner/scattering code.
- **`muda/`** — Vendored header-only CUDA abstraction library providing device buffers, linear algebra views, viewers, launch helpers, CUB wrappers, and compute-graph utilities.

---

## Build Instructions

### Prerequisites

- NVIDIA GPU with compute capability supported by your CUDA toolkit
- CMake ≥ 3.18
- CUDA toolkit ≥ 11.0
- vcpkg (recommended on Windows/Linux) or system packages providing:
  - `eigen3` 3.4.0
  - `freeglut` 3.4.0
  - `glew` 2.2.0
  - `nlohmann-json`
- On Linux you can alternatively install with apt:
  ```bash
  sudo apt install libglew-dev freeglut3-dev libeigen3-dev nlohmann-json3-dev
  ```

### Configure and build

From the repository root:

```bash
# If using vcpkg, set the toolchain (only needed once per shell)
export CMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake

# Configure
cmake -B build -S .

# Build
cmake --build build --config Release
```

The produced executable is `build/gipc` (or `build/Release/gipc.exe` on Windows). The CMake build type defaults to `Release` and on Linux adds `-O3`.

### Important build notes

- The project is built as a single executable target named `gipc`. All `.cu`/`.cpp` files under `StiffGIPC/` are globbed by CMake.
- CUDA separable compilation is enabled (`CUDA_SEPARABLE_COMPILATION ON`).
- Default CUDA architecture is `native`. If CMake cannot detect it, set `CMAKE_CUDA_ARCHITECTURES` explicitly, e.g. `-DCMAKE_CUDA_ARCHITECTURES=86`.
- For CUDA ≥ 13, the build adds `${CUDAToolkit_INCLUDE_DIRS}/cccl` to include paths.
- `compile_commands.json` is generated automatically (`CMAKE_EXPORT_COMPILE_COMMANDS ON`).
- Two compile definitions bake absolute paths into the binary:
  - `GIPC_ASSETS_DIR="<repo>/Assets/"`
  - `GIPC_OUTPUT_DIR="<repo>/Output/"`
  The executable therefore expects to be run from/relative to the repository root, and the `Output/` directory is used for saved meshes/screenshots.

---

## Running the Program

The executable uses hard-coded scene loading in `gl_main.cu`. It looks for files under `Assets/` and reads:

- `Assets/scene/parameterSetting.txt` — global simulation parameters (time step, friction, Young's moduli, solver tolerances, etc.).
- Scene JSON files under `Assets/scene/json/` — per-object rigid-body descriptions.
- Mesh files under `Assets/tetMesh/` and `Assets/triMesh/`.
- Optional `Assets/scene/abd_system_config.json` — extra ABD parameters (`motor_speed`, `motor_strength`).

At runtime the GLUT window accepts keyboard/mouse input (defined in `gl_main.cu`). Common operations include stepping the simulation, saving the surface mesh, and taking screenshots. The code outputs `.obj` surface meshes to `Output/` when surface saving is enabled.

---

## Code Style and Conventions

### Formatting

A `.clang-format` file is present at the repository root. Key choices:

- Based on LLVM style
- Column limit: 80
- Indent: 4 spaces, no tabs
- Pointer alignment: left (`Float* ptr`)
- Braces on new lines for classes, functions, namespaces, control statements
- Short functions are allowed on a single line only when inline
- Sort includes: disabled

Run formatting with:

```bash
clang-format -i StiffGIPC/path/to/file.cu StiffGIPC/path/to/file.cuh
```

### Naming conventions observed in the code

- `PascalCase` for classes (`GIPC`, `ABDSystem`, `GlobalLinearSystem`).
- `snake_case` for functions and member variables (`init_system`, `compute_energy`).
- `m_` prefix for private member variables.
- `gipc::` namespace wraps most project code; `muda::` is the CUDA utility namespace.
- CUDA kernel files use `.cu` for implementation and `.cuh` for headers that contain device code.
- Type aliases for vectors/matrices are centralized in `gipc/type_define.h` (`Vector3`, `Matrix12x12`, `Float = double`, etc.).
- Device-buffer types from muda are used heavily (`muda::DeviceBuffer<T>`, `muda::DeviceVar<T>`, `muda::DeviceDenseVector<T>`).

### Code organization patterns

- Headers are included with angle brackets relative to `StiffGIPC/` (e.g. `#include <gipc/type_define.h>`).
- Large classes split compute-heavy methods into `.cu` implementation files while keeping declarations in `.h`/`.cuh`.
- Template/inline device code is placed in `.inl` files inside `details/` subdirectories and included from the main header.
- The legacy GIPC core uses raw device pointers and Thrust; newer ABD/linear-system code uses `muda` buffer and linear-system abstractions.

---

## Testing

There is **no automated test suite** in this repository. `muda/` contains an external `catch2` header and its own CMake test/example options, but the StiffGIPC application itself has no tests.

### Validation workflow

- Build the project in `Release` mode.
- Run `gipc` and load a provided scene (e.g. `wrecking-ball-simple.json`).
- Verify that the simulation progresses without NaNs/crashes and that output meshes are written to `Output/`.
- Compare outputs/visuals against expected behavior from the paper/video.

When modifying numerical code, prefer testing on the small included scenes first and watch the `Output/` folder for NaN/invalid vertices.

---

## Mesh Preprocessing (MeshProcess)

Before running a high-resolution FEM scene, tetrahedral meshes may need to be partitioned with METIS. The `MeshProcess/metis_partition` tool does this offline.

Typical usage (from the repository root after building):

```bash
./build/MeshProcess/metis_partition/metis_partition <input.msh> <nparts>
```

This produces a sorted `.msh`/`.obj` and a `.part` file in `Assets/sorted_mesh/`. The `SimpleSceneImporter` and `load_mesh` code consume these `.part` files when the preconditioner type is non-zero.

---

## Common Hazards and Development Notes

- **No `std::filesystem` path portability in GLUT paths**: asset paths are baked as compile definitions; moving the binary without the repository tree will break loading.
- **Memory**: the `GIPC` class allocates large device buffers. Expanding resolution or collision buffers may require editing constants/reserve sizes in `GIPC.cuh`/`.cu`.
- **CUDA architecture mismatch**: if you get "no kernel image" errors, set `CMAKE_CUDA_ARCHITECTURES` to match your GPU.
- **Feature flags are global**: toggling `USE_FRICTION`, `USE_SNK1`, etc. requires reconfiguring CMake because they are `target_compile_definitions`.
- **Mixed coding styles**: the repository combines legacy raw-pointer GIPC code with newer muda-based ABD/linear-system code. When adding features, follow the style of the module you are touching.
- **Visualization is not optional**: `gl_main.cu` links GLUT/GLEW and creates a window; there is currently no headless batch-mode entry point.
- **No CI/CD**: build verification and scene testing are manual.

---

## Useful Files for Agents

| Task | Files to read first |
|------|---------------------|
| Build / dependencies | `CMakeLists.txt`, `MeshProcess/CMakeLists.txt`, `StiffGIPC/muda/CMakeLists.txt` |
| Entry point / runtime flow | `StiffGIPC/gl_main.cu` |
| Core IPC physics | `StiffGIPC/GIPC.cu`, `StiffGIPC/GIPC.cuh` |
| Affine body dynamics | `StiffGIPC/abd_system/abd_system.h`, `StiffGIPC/abd_system/abd_sim_data.h` |
| Linear solver | `StiffGIPC/linear_system/linear_system/global_linear_system.h`, `StiffGIPC/linear_system/solver/pcg_solver.h` |
| Collision detection | `StiffGIPC/mlbvh.cuh`, `StiffGIPC/ACCD.cuh` |
| Types/aliases | `StiffGIPC/gipc/type_define.h` |
| Scene loading | `StiffGIPC/gipc/utils/simple_scene_importer.h` |
| Global parameters | `Assets/scene/parameterSetting.txt` |

---

## Summary for Quick Orientation

StiffGIPC is a single-CMake, C++/CUDA research application. It has no tests and no package manager manifest beyond CMake `find_package` calls. The build depends on CUDA, Eigen, FreeGLUT/GLEW, OpenGL, and nlohmann-json. The executable loads scene files from `Assets/`, runs a GPU IPC simulation loop, and visualizes with OpenGL. When editing, respect the existing mixed legacy/muda style, keep changes minimal, and validate by rebuilding and running a bundled scene.
