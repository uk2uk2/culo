# culo
(CUDA Lua Orchestrator) - a PoC hotswapper script for C/C++ and CUDA kernel deployment


How It Works
The C/CUDA Code:

Implements two CUDA kernels and two corresponding wrapper functions (exposed to Lua).
Registers these functions in a Lua module called "myalgos".
Creates a Lua state, loads standard libraries, registers the module, and then runs a provided Lua script.
The Lua Script:

Loads the "myalgos" module.
Calls dotProductNaive() and dotProductOptimized(), printing their results.
In a more complete system, the Lua code might choose between algorithms based on runtime data or configuration.
Compilation & Execution:

Compile using NVCC (ensuring the Lua libraries are linked).
Run the resulting executable and pass the Lua script as an argument:
bash
Copy
./orchestrator orchestrator.lua


This example is kept small and self‑contained for a proof of concept
