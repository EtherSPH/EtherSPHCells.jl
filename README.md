# EtherSPHCells.jl

An improved version on cell-based neighbour search.

Largely thanks to [SmoothedParticles.jl](https://github.com/OndrejKincl/SmoothedParticles.jl). I almost copy all the codes from his repository. **Please cite his code** in any case you use this repository's code.

# Interesting things I find

## Harmful runtime dispatch in julia

As an import feature in julia, multi-dispatch is of good use. However, runtime dispatch in julia is harmful to performance.

## Best threads usage in julia

In my previous practice, `FLoops` package does better in multi-threads work. However, this time I find `Threads.@threads` is better for `FLoops.@floop @simd`.