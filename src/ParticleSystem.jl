#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/21 00:44:17
  @ license: MIT
  @ description:
 =#

# * default AbstractParticle must at least have fields:
# * - x_vec_::RealVector
# * - rho_::Float64
# * - mass_::Float64
# * - type_::Int64
abstract type AbstractParticle end

@inline function distance(p::AbstractParticle, q::AbstractParticle)::Float64
    return norm(p.x_vec_ - q.x_vec_)
end

@inline function noneFunction!(p::AbstractParticle)::Nothing
    return nothing
end

mutable struct ParticleSystem{ParticleType <: AbstractParticle}
    particles_::Vector{ParticleType}
    background_cell_list_::BackgroundCellList
end

function ParticleSystem(particle_type::DataType, background_cell_list::BackgroundCellList)::ParticleSystem
    return ParticleSystem(Vector{particle_type}(), background_cell_list)
end

function ParticleSystem(
    particle_type::DataType,
    reference_smoothing_radius::Float64,
    calculation_domain_bounding_box::CalculationDomainBoundingBox,
    restrict_calculation_domain::Shape,
)::ParticleSystem
    background_cell_list =
        BackgroundCellList(reference_smoothing_radius, calculation_domain_bounding_box, restrict_calculation_domain)
    return ParticleSystem(particle_type, background_cell_list)
end

function ParticleSystem(
    particle_type::DataType,
    reference_smoothing_radius::Float64,
    calculation_domain_bounding_box::CalculationDomainBoundingBox,
)::ParticleSystem
    background_cell_list = BackgroundCellList(reference_smoothing_radius, calculation_domain_bounding_box)
    return ParticleSystem(particle_type, background_cell_list)
end

function ParticleSystem(
    particle_type::DataType,
    reference_smoothing_radius::Float64,
    start_point::RealVector,
    end_point::RealVector,
)::ParticleSystem
    calculation_domain_bounding_box = CalculationDomainBoundingBox(start_point, end_point)
    return ParticleSystem(particle_type, reference_smoothing_radius, calculation_domain_bounding_box)
end

Base.show(io::IO, p::ParticleSystem) = print(
    io,
    "ParticleSystem:\n",
    "    with particle type $(getParticleType(p))\n",
    "    and $(length(p.particles_)) particles\n",
    "   in background cell list $(p.background_cell_list_)",
)

@inline getParticleType(::ParticleSystem{ParticleType}) where {ParticleType} = ParticleType

function updateBackgroundCellList!(particle_system::ParticleSystem)::Nothing
    # * 1. clear all the particles' id marked in the background cell list
    Threads.@threads for cell in particle_system.background_cell_list_.cell_list_
        for i in eachindex(cell.contained_particle_global_ids_)
            @inbounds cell.contained_particle_global_ids_[i] = 0
        end
    end
    # * reset removal cell
    for i in eachindex(particle_system.background_cell_list_.to_be_removed_cell_.contained_particle_global_ids_)
        @inbounds particle_system.background_cell_list_.to_be_removed_cell_.contained_particle_global_ids_[i] = 0
    end
    # * 2. we need first add particles to the to_be_removed_cell_
    # * then reset particles
    # * for reset particles will change their global ids
    Threads.@threads for particle_global_id in eachindex(particle_system.particles_)
        particle_position = particle_system.particles_[particle_global_id].x_vec_
        if !isInBackgroundCellList(particle_position, particle_system.background_cell_list_)
            addParticleToCell!(particle_system.background_cell_list_.to_be_removed_cell_, particle_global_id)
        end
    end
    # * reset particles
    the_first_zero_index = 1
    for i in eachindex(particle_system.background_cell_list_.to_be_removed_cell_.contained_particle_global_ids_)
        @inbounds if particle_system.background_cell_list_.to_be_removed_cell_.contained_particle_global_ids_[i] == 0
            the_first_zero_index = i
            break
        else
            the_first_zero_index = i + 1
        end
    end
    to_be_removed_ids =
        particle_system.background_cell_list_.to_be_removed_cell_.contained_particle_global_ids_[1:(the_first_zero_index - 1)]
    sort!(to_be_removed_ids)
    deleteat!(particle_system.particles_, to_be_removed_ids)
    # * 3. add particles to the background cell list
    Threads.@threads for particle_global_id in eachindex(particle_system.particles_)
        global_index = getGlobalIndexInBackgroundCellList(
            particle_system.particles_[particle_global_id].x_vec_,
            particle_system.background_cell_list_,
        )
        addParticleToCell!(particle_system.background_cell_list_.cell_list_[global_index], particle_global_id)
    end
    return nothing
end

"""
    interactionFunction!(p::AbstractParticle, q::AbstractParticle, r::Float64; parameters...)::Nothing
"""
@inline function applyInteractionFunction!(
    particle_system::ParticleSystem,
    interactionFunction!::Function,
    p::AbstractParticle,
    p_id::Int64;
    parameters...,
)::Nothing
    cell_id = getGlobalIndexInBackgroundCellList(p.x_vec_, particle_system.background_cell_list_)
    @inbounds for i_neighbour_cell in
                  particle_system.background_cell_list_.cell_list_[cell_id].neighbour_cell_global_ids_
        cell = particle_system.background_cell_list_.cell_list_[i_neighbour_cell]
        for q_id in cell.contained_particle_global_ids_
            if q_id == 0
                break
            end
            if p_id == q_id
                continue
            end
            q = particle_system.particles_[q_id]
            r = distance(p, q)
            if r > particle_system.background_cell_list_.reference_smoothing_radius_
                continue
            end
            interactionFunction!(p, q, r; parameters...)
        end
    end
    return nothing
end

@inline function applyInteraction!(
    particle_system::ParticleSystem,
    interactionFunction!::Function;
    parameters...,
)::Nothing
    Threads.@threads for particle_global_id in eachindex(particle_system.particles_)
        @inbounds applyInteractionFunction!(
            particle_system,
            interactionFunction!,
            particle_system.particles_[particle_global_id],
            particle_global_id;
            parameters...,
        )
    end
    return nothing
end

@inline function applySelfaction!(
    particle_system::ParticleSystem,
    selfactionFunction!::Function;
    parameters...,
)::Nothing
    Threads.@threads for particle_global_id in eachindex(particle_system.particles_)
        @inbounds selfactionFunction!(particle_system.particles_[particle_global_id]; parameters...)
    end
    return nothing
end
