#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/21 17:56:22
  @ license: MIT
  @ description: use `.vtp` for output, and add some helper functions
 =#

const kFileExtension = ".vtp"
const kWallTimeFormat = "yyyy_mm_dd_HH_MM_SS.SS"
const kDensityString = "Density"
const kMassString = "Mass"
const kTypeString = "Type"
const kVelocityString = "Velocity"

@inline function assurePathExist(path::String)::Nothing
    if !isdir(path)
        @info "Create directory: $path"
        mkpath(path)
    else
        # remove all files in the directory
        @info "Remove all files in directory: $path"
        for file in readdir(path)
            rm(joinpath(path, file))
        end
    end
    return nothing
end

@inline function getWallTime()::String
    return Dates.format(now(), kWallTimeFormat)
end

@inline getType(particle::AbstractParticle)::Int64 = particle.type_
@inline getMass(particle::AbstractParticle)::Float64 = particle.mass_
@inline getDensity(particle::AbstractParticle)::Float64 = particle.rho_
@inline getPosition(particle::AbstractParticle)::RealVector = particle.x_vec_
@inline getVelocity(particle::AbstractParticle)::RealVector = particle.v_vec_

@kwdef mutable struct VTPIO
    output_count_::Int64 = 0
    step_digit_::Int64 = 3
    file_name_::String = "result"
    output_path_::String = "example/results"
    scalar_name_list_::Vector{String} = String[kMassString, kDensityString]
    getScalarFunctions_::Vector{Function} = Function[getMass, getDensity]
    vector_name_list_::Vector{String} = String[]
    getVectorFunctions_::Vector{Function} = Function[]
end

Base.show(io::IO, vtp_io::VTPIO) = print(
    io,
    "VTPIO:\n",
    "    step digit: $(vtp_io.step_digit_)\n",
    "    file name: $(vtp_io.file_name_)\n",
    "    output path: $(vtp_io.output_path_)\n",
    "    scalar name list: $(vtp_io.scalar_name_list_)\n",
    "    vector name list: $(vtp_io.vector_name_list_)\n",
    "    current writen time: $(vtp_io.output_count_)\n",
)

@inline assurePathExist(vtp_io::VTPIO)::Nothing = assurePathExist(vtp_io.output_path_)

@inline function getFileName(vtp_io::VTPIO)::String
    return joinpath(
        vtp_io.output_path_,
        string(vtp_io.file_name_, string(vtp_io.output_count_, pad = vtp_io.step_digit_), kFileExtension),
    )
end

@inline function addScalar!(vtp_io::VTPIO, scalar_name::String, scalarFunction::Function)::Nothing
    push!(vtp_io.scalar_name_list_, scalar_name)
    push!(vtp_io.getScalarFunctions_, scalarFunction)
    return nothing
end

@inline function addVector!(vtp_io::VTPIO, vector_name::String, vectorFunction::Function)::Nothing
    push!(vtp_io.vector_name_list_, vector_name)
    push!(vtp_io.getVectorFunctions_, vectorFunction)
    return nothing
end

@inline function saveVTP(vtp_io::VTPIO, particle_system::ParticleSystem, step::Int64, simulation_time::Float64)::Nothing
    n_particles = length(particle_system.particles_)
    type = zeros(Int64, n_particles)
    positions = zeros(Float64, 3, n_particles)
    scalars_list = [zeros(Float64, n_particles) for _ in 1:length(vtp_io.scalar_name_list_)]
    vectors_list = [zeros(Float64, 3, n_particles) for _ in 1:length(vtp_io.vector_name_list_)]
    Threads.@threads for i in eachindex(particle_system.particles_)
        @inbounds type[i] = getType(particle_system.particles_[i])
        @inbounds positions[:, i] = getPosition(particle_system.particles_[i])
        for j in eachindex(vtp_io.scalar_name_list_)
            @inbounds scalars_list[j][i] = vtp_io.getScalarFunctions_[j](particle_system.particles_[i])
        end
        for j in eachindex(vtp_io.vector_name_list_)
            @inbounds vectors_list[j][:, i] = vtp_io.getVectorFunctions_[j](particle_system.particles_[i])
        end
    end
    file_name = getFileName(vtp_io)
    cells = [MeshCell(PolyData.Verts(), [i]) for i in 1:n_particles]
    vtp_file = vtk_grid(file_name, positions, cells)
    vtp_file["TMSTEP"] = step
    vtp_file["TimeValue"] = simulation_time
    vtp_file["WallTime"] = getWallTime()
    vtp_file["Type"] = type
    for i in eachindex(vtp_io.scalar_name_list_)
        @inbounds vtp_file[vtp_io.scalar_name_list_[i]] = scalars_list[i]
    end
    for i in eachindex(vtp_io.vector_name_list_)
        @inbounds vtp_file[vtp_io.vector_name_list_[i]] = vectors_list[i]
    end
    vtk_save(vtp_file)
    vtp_io.output_count_ += 1
    return nothing
end
