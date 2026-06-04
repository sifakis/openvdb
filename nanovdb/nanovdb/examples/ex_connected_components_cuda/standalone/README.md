# Standalone tools

Self-contained helper programs for the connected-components work. These are **not**
part of the `ex_connected_components_cuda` build: the `nanovdb_example` CMake helper
globs `*.cpp`/`*.cu` directly in the example root (non-recursively), so anything in
this subdirectory is intentionally excluded — each tool here has its own `main()` and
is meant to be compiled by hand.

## `cc_vis.cpp`

A pedagogical visualizer for the Shiloach-Vishkin connected-components primitives on a
2D 8x8 grid. Pick an alive-cell pattern (`full` / `snake` / `band` / `net`) and step
through `Propagate`, `SV-Hook`, and `Compress` one operation at a time, printing the
parent/label grid after each step. Pure host C++ (no NanoVDB / CUDA / OpenVDB).

It mirrors, in 2D and on the CPU, the same SV union-find that
`nanovdb::tools::cuda::ConnectedComponents::processLeafConnectedComponents()` runs per
8^3 leaf on the device — a quick way to build intuition for the hook/compress schedule.

```bash
g++ -std=c++17 -O2 cc_vis.cpp -o cc_vis
./cc_vis [pattern]      # pattern optional; omitted -> interactive prompt
```
