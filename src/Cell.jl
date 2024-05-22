#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/19 15:20:45
  @ license: MIT
  @ description:
 =#

mutable struct Cell
    contained_particle_global_ids_::Vector{Int64}
    thread_lock_::Base.Threads.ReentrantLock
    neighbour_cell_global_ids_::Vector{Int64}
    Cell() = new(Vector{Int64}(), Base.Threads.ReentrantLock(), Vector{Int64}())
end

Base.length(cell::Cell) = length(cell.contained_particle_global_ids_)
Base.size(cell::Cell) = (length(cell.contained_particle_global_ids_),)
Base.IndexStyle(::Type{<:Cell}) = Base.IndexLinear()
Base.getindex(cell::Cell, i::Int64) = getindex(cell.contained_particle_global_ids_, i)
Base.setindex!(cell::Cell, value::Int64, i::Int64) = setindex!(cell.contained_particle_global_ids_, value, i)
Base.show(io::IO, cell::Cell) = print(
    io,
    "Cell:\n",
    "    the cell contain particles' ids: $(cell.contained_particle_global_ids_)\n",
    "    neighbour cell global ids: $(cell.neighbour_cell_global_ids_)\n",
)

@inline function latestIndexToInsertParticle!(cell::Cell)::Int64
    @inbounds @simd for i in eachindex(cell.contained_particle_global_ids_)
        if cell.contained_particle_global_ids_[i] == 0
            return i
        end
    end
    new_length = length(cell.contained_particle_global_ids_) + 1
    resize!(cell.contained_particle_global_ids_, new_length)
    return new_length
end

@inline function addParticleToCell!(cell::Cell, particle_global_id::Int64)::Nothing
    Base.Threads.lock(cell.thread_lock_)
    try
        latest_index = latestIndexToInsertParticle!(cell)
        cell[latest_index] = particle_global_id
        # reorder the particles' ids here
        @inbounds while latest_index > 1 && cell[latest_index - 1] < cell[latest_index]
            swap_index = cell[latest_index]
            cell[latest_index] = cell[latest_index - 1]
            cell[latest_index - 1] = swap_index
            latest_index -= 1
        end
    finally
        Base.Threads.unlock(cell.thread_lock_)
    end
    return nothing
end
