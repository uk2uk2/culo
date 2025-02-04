-- orchestrator.lua
print("=== Hotswap Algorithm Orchestration PoC ===")

-- Require the module that contains our CUDA-based algorithms.
local myalgos = require("myalgos")

print("Running naive dot product:")
local result_naive = myalgos.dotProductNaive()
print("Naive result: ", result_naive)

print("Running optimized dot product:")
local result_optimized = myalgos.dotProductOptimized()
print("Optimized result: ", result_optimized)
