#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/20 21:20:09
  @ license: MIT
  @ description:
 =#

using Test
using EtherSPHCells
using JuliaFormatter

JuliaFormatter.format(
    ".",
    indent = 4,
    margin = 120,
    always_for_in = true,
    whitespace_typedefs = true,
    whitespace_ops_in_indices = true,
    remove_extra_newlines = true,
    pipe_to_function_call = false,
    always_use_return = true,
    whitespace_in_kwargs = true,
    trailing_comma = true,
)

@testset "EtherSPHCells" begin
    @testset "CartesianIndex.jl" begin
        include("CartesianIndexTest.jl")
    end
end
