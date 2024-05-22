#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/18 22:22:36
  @ license: MIT
  @ description:
 =#

module EtherSPHCells

using StaticArrays
using WriteVTK
using Dates
using Parameters
using JuliaFormatter
using ExportAll

include("MathBase.jl")
include("SmoothKernel.jl")
include("CartesianIndex.jl")
include("Geometry.jl")
include("Cell.jl")
include("BackgroundCellList.jl")
include("ParticleSystem.jl") # core file
include("PostProcess.jl")

JuliaFormatter.format(
    ".",
    indent = 4, # 4 spaces for each indent
    margin = 120, # 120 characters for each line
    always_for_in = true, # always use `for i in 1:10` instead of `for i = 1:10`
    whitespace_typedefs = true, # add whitespace around `<:`
    whitespace_ops_in_indices = true, # add whitespace around `+`, `-`, `*`, `/` in indices
    remove_extra_newlines = true, # remove extra newlines
    pipe_to_function_call = false, # add whitespace around `|>` in function call
    always_use_return = true, # always use `return` in function
    whitespace_in_kwargs = true, # add whitespace around `=` in kwargs
    trailing_comma = true, # add trailing comma
)

ExportAll.@exportAll()

end # module EtherSPHCells
