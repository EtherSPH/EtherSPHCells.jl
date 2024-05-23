# EtherSPHCells.jl

An improved version on cell-based neighbour search.

Largely thanks to [SmoothedParticles.jl](https://github.com/OndrejKincl/SmoothedParticles.jl). I almost copy all the codes from his repository. **Please cite his code** in any case you use this repository's code.

For my personal research need, a deep insight into SPH method is a must. So I began to search open-source SPH code. `SPHinxsys` is a good choice but the hardness on `c++` discouraged me. Later I found [SmoothedParticles.jl](https://github.com/OndrejKincl/SmoothedParticles.jl), which is easy to understand and has good performance.

I learned a lot from it and develop my own package. For further work, an `ActionLibrary.jl` is added to include any particle action I've got in paper and books.

## How to start

### Install `julia` and related packages to run demo or example

First make sure you have julia installed on your pc. Then in julia terminal, change the current working directory to root, type `]` to `pkg` model.

```bash
julia> ]
```

Then add the packages:

```bash
pkg> activate .
pkg> instantiate
```

Now you can run files in `example` or `demo` like:

```bash
julia> ]
pkg> activate .
julia> include("example/xxx/xxx.jl)
julia> main()
```

### Learn to use

You may see the file: `demo/collapse_dry/collapse_dry_base.jl` and `demo/collapse_dry/collapse_dry.jl`. With detailed comments, you can learn how to develop a case for your SPH model.

### Action Library

Based on [SmoothedParticles.jl](https://github.com/OndrejKincl/SmoothedParticles.jl), I add some template particle action in `ActionLibrary.jl` enabling user adopt them in their cases.

## Interesting things I find

### Harmful runtime dispatch

As an import feature in julia, multi-dispatch is in widely use. However, runtime dispatch in julia is harmful to performance.

### Best threads usage

In my previous practice, `FLoops` package does better in multi-threads work. However, this time I find `Threads.@threads` is better for `FLoops.@floop @simd`.