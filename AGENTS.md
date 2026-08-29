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
| GPU compute | NVIDIA CUDA ≥ 11.0 | Uses CUB, cuSPARSE, cuBLAS, cuSOLVER; active code does not use Thrust |
| Math/linear algebra | Eigen 3.4.0 | Host-side dense math; also used inside CUDA via cuda_tools |
| Visualization | FreeGLUT 3.4.0 + GLEW 2.2.0 + OpenGL | Interactive viewer in `gl_main.cu` |
| JSON | nlohmann/json | Scene and config files |
| Partitioning | METIS + GKlib | Preprocess meshes for domain decomposition (in `MeshProcess/`) |
| CUDA wrapper library | cuda_tools | In-repo CUDA utility layer under `StiffGIPC/cuda_tools/` (the formerly vendored muda has been removed) |

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
└── StiffGIPC/                  # Main simulation source code (include root for `<...>` paths)
    ├── app/
    │   └── gl_main.cu          # Entry point: GLUT window, rendering, user input
    ├── core/                   # Core IPC system
    │   ├── GIPC.cu / GIPC.cuh          # Main GIPC class: device state and per-step logic
    │   ├── GIPC_PDerivative.cuh        # IPC barrier derivative kernels
    │   ├── gipc_system.cu              # GIPC methods wiring ABD + linear-system assembly
    │   ├── body_boundary_type.h
    │   └── gipc_path.h                 # Baked asset/output path helpers
    ├── collision/              # Collision detection & friction
    │   ├── ACCD.cu / ACCD.cuh          # Adaptive Continuous Collision Detection
    │   ├── mlbvh.cu / mlbvh.cuh        # Linear BVH broad-phase
    │   └── FrictionUtils.cuh           # Friction-related CUDA helpers
    ├── fem/                    # FEM deformable model
    │   ├── femEnergy.cu/.cuh           # FEM elasticity energy/gradient/Hessian
    │   ├── device_fem_data.cu/.cuh     # Device FEM mesh/state representation
    │   └── fem_parameters.h
    ├── solver/                 # Legacy solver stack still in active use
    │   ├── PCG_SOLVER.cu/.cuh          # PCG_Data held by GIPC (pcg_data member)
    │   └── MASPreconditioner.cu/.cuh   # MAS preconditioner (used when P_type == 1)
    ├── math/                   # GPU math helpers
    │   ├── gpu_eigen_libs.cu/.cuh      # __GEIGEN__ matrix/vector device math
    │   ├── eigen_data.h
    │   ├── QRSVD.hpp / givens.hpp      # 3x3 SVD via QR
    ├── io/
    │   └── load_mesh.cpp/.h    # Host-side mesh I/O (OBJ/MSH)
    ├── abd_system/             # Affine body dynamics system
    ├── linear_system/          # Global linear system assembly + PCG solver + preconditioners
    ├── cuda_tools/             # In-repo CUDA utility layer (device buffers, views, CUB wrappers)
    └── gipc/                   # Common types, statistics, timer, scene importer, utilities
```

### Module responsibilities

- **`app/gl_main.cu`** — Creates the OpenGL window, loads meshes via `SimpleSceneImporter`, initializes CUDA, builds the IPC system, and runs the per-frame simulation loop. It is also the only file with a `main()`.
- **`core/GIPC.cu/.cuh`** — The main `GIPC` class holding device pointers for vertices, faces, edges, collision pairs, barrier/friction data, and the global triplet Hessian. Implements barrier energy, gradients, Hessians, line search, CCD, and Newton step logic. `core/gipc_system.cu` holds the `GIPC` methods that wire up the ABD system and the global linear system.
- **`abd_system/`** — `ABDSystem` and `ABDSimData`: affine-body state (`q`, `q_tilde`, `q_prev`, `q_v`), Jacobians, dyadic mass, gravity, shape/kinetic energy, and system assembly. Contains the math that maps between affine DOFs and world positions.
- **`linear_system/`** — `GlobalLinearSystem` assembles FEM + ABD subsystems (`fem_linear_subsystem`, `abd_linear_subsystem`) into a global matrix and solves it with a PCG solver (`pcg_solver`) and preconditioners (diagonal, ABD block-diagonal, FEM mass).
- **`solver/`** — Legacy solver stack that is still actively used: `GIPC` owns a `PCG_Data pcg_data` member (defined in `PCG_SOLVER.cuh`), and its `MASPreconditioner MP` is passed into the new `MAS_Preconditioner` when `P_type == 1`. Do not delete; the two solver stacks are coupled.
- **`collision/mlbvh.cu/.cuh`** — Broad-phase collision detection using a linear BVH on the GPU.
- **`collision/ACCD.cu/.cuh`** — Narrow-phase continuous collision detection and largest-feasible-step-size queries.
- **`MeshProcess/`** — Offline preprocessing: takes a tetrahedral mesh and writes a METIS partition file used by the FEM preconditioner/scattering code.
- **`cuda_tools/`** — In-repo CUDA utility layer (namespace `cudatool`) providing device buffers, buffer views, dense vectors, CUB wrappers, atomics and debug helpers. It replaced the formerly vendored `muda` library. `DeviceBuffer<T>` (in `cuda_buffer_view.h`) is the single owning buffer class used project-wide for device memory. `resize()` preserves the old logical range; when capacity is exceeded, `resize_discard()`, `resize_preserve()`, and `reserve_amortized()` allocate 150% of the latest requirement, including sudden jumps beyond the old capacity. `reserve()`/`reserve_amortized()` never change logical size. `DeviceBuffer` is move-only; use `copy_from()` for an intentional device-to-device value copy, and use `.data()`/a view when legacy code previously copied a raw pointer with `auto`. The implicit `operator T*()` exists only for legacy call sites; never cache it across a possible resize and never `cudaFree` it directly.

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

For non-interactive validation, set `GIPC_STEPS` to a positive frame count. The program hides its GLUT window, advances exactly that many frames, then exits:

```powershell
$env:GIPC_STEPS = "1"
.\build\Release\gipc.exe
```

Set `GIPC_CASE=1` through `6` to select `set_case1()` through `set_case6()` without editing source. Set `GIPC_DUMP_STATE` to a file path to write the final device vertex array as packed binary `double3` records for numerical regression comparisons.

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
- `gipc::` namespace wraps most project code; `cudatool::` is the CUDA utility namespace.
- CUDA kernel files use `.cu` for implementation and `.cuh` for headers that contain device code.
- Type aliases for vectors/matrices are centralized in `gipc/type_define.h` (`Vector3`, `Matrix12x12`, `Float = double`, etc.).
- Device-buffer types from cuda_tools are used heavily (`cudatool::DeviceBuffer<T>`, `cudatool::BufferView<T>`, `cudatool::DenseVectorView<T>`).

### Code organization patterns

- Headers are included with angle brackets relative to `StiffGIPC/` (e.g. `#include <gipc/type_define.h>`).
- Large classes split compute-heavy methods into `.cu` implementation files while keeping declarations in `.h`/`.cuh`.
- Template/inline device code is placed in `.inl` files inside `details/` subdirectories and included from the main header.
- The legacy GIPC core still has raw-pointer interfaces; active scans, sorts, reductions, and segmented reductions use CUB through persistent `cuda_tools` workspaces.

---

## Testing

`dynamic_memory_tests` exercises zero-capacity device buffers, repeated preserving growth, collision count/grow/rerun, independent full-CCD growth, global-triplet workspace growth, contact partitions that start at zero, partial-warp segmented reduction, and persistent CUB scratch.

```bash
cmake --build build --config Release --target dynamic_memory_tests
ctest --test-dir build -C Release --output-on-failure
```

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
- **Memory**: DCD pair/index buffers start with capacity 100,000 and the independent CCD buffer starts with capacity 1,000,000; logical sizes remain zero until detection. Global-Hessian live capacity starts from the exact fixed-energy triplet count plus `100000 * M12_Off` PT blocks. Values/rows/cols reserve twice that live estimate for disjoint conversion staging, while hash/index scratch reserves the live estimate. Whenever any dynamic requirement exceeds capacity, the replacement allocation is 150% of that latest requirement, not 150% of the old capacity. Overflow still uses guarded count/grow/re-run semantics. BVH and MAS receive current output pointers at each call rather than caching reallocatable addresses. Temporary friction/constraint buffers and per-thread CUB scratch retain capacity after warm-up.
- **CUDA architecture mismatch**: if you get "no kernel image" errors, set `CMAKE_CUDA_ARCHITECTURES` to match your GPU.
- **Feature flags are global**: toggling `USE_FRICTION`, `USE_SNK1`, etc. requires reconfiguring CMake because they are `target_compile_definitions`.
- **Mixed coding styles**: the repository combines legacy raw-pointer GIPC code with newer cuda_tools-based ABD/linear-system code. When adding features, follow the style of the module you are touching.
- **Visualization remains linked in batch mode**: `GIPC_STEPS` hides the GLUT window and provides deterministic finite-frame execution, but a working OpenGL context is still required.
- **No CI/CD**: build verification and scene testing are manual.

---

## Useful Files for Agents

| Task | Files to read first |
|------|---------------------|
| Build / dependencies | `CMakeLists.txt`, `MeshProcess/CMakeLists.txt` |
| Entry point / runtime flow | `StiffGIPC/app/gl_main.cu` |
| Core IPC physics | `StiffGIPC/core/GIPC.cu`, `StiffGIPC/core/GIPC.cuh` |
| Affine body dynamics | `StiffGIPC/abd_system/abd_system.h`, `StiffGIPC/abd_system/abd_sim_data.h` |
| Linear solver | `StiffGIPC/linear_system/linear_system/global_linear_system.h`, `StiffGIPC/linear_system/solver/pcg_solver.h` |
| Legacy solver / MAS preconditioner | `StiffGIPC/solver/PCG_SOLVER.cuh`, `StiffGIPC/solver/MASPreconditioner.cuh` |
| Collision detection | `StiffGIPC/collision/mlbvh.cuh`, `StiffGIPC/collision/ACCD.cuh` |
| FEM model | `StiffGIPC/fem/femEnergy.cuh`, `StiffGIPC/fem/device_fem_data.cuh` |
| Types/aliases | `StiffGIPC/gipc/type_define.h` |
| Scene loading | `StiffGIPC/gipc/utils/simple_scene_importer.h` |
| Global parameters | `Assets/scene/parameterSetting.txt` |

---

## Summary for Quick Orientation

StiffGIPC is a single-CMake, C++/CUDA research application. It has no tests and no package manager manifest beyond CMake `find_package` calls. The build depends on CUDA, Eigen, FreeGLUT/GLEW, OpenGL, and nlohmann-json. The executable loads scene files from `Assets/`, runs a GPU IPC simulation loop, and visualizes with OpenGL. When editing, respect the existing mixed legacy/cuda_tools style, keep changes minimal, and validate by rebuilding and running a bundled scene.
