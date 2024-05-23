#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/23 17:57:47
  @ license: MIT
  @ description:
 =#

using EtherSPHCells
using Parameters
using ProgressBars

const dim = 2
const dr = 0.002
const gap = dr
const h = 3 * dr

const smooth_kernel = SmoothKernel(h, dim, CubicSpline)

const water_width = 0.114
const water_height = 0.114
const box_width = 0.42
const box_height = 0.44
const wall_width = h

const rho_0 = 1000.0
const mass = rho_0 * dr^dim
const gravity = 9.8
const c = 10 * sqrt(2 * gravity * water_height)
const g = RealVector(0.0, -gravity, 0.0)
const mu = 1e-3
const nu = mu / rho_0

@inline function getPressureFromDensity(rho::Float64)::Float64
    return c^2 * (rho - rho_0)
end

@inline function getDensityFromPressure(p::Float64)::Float64
    return p / c^2 + rho_0
end

const dt = 0.1 * h / c
const t_end = 3.0
const output_dt = 100 * dt
const density_filter_dt = 20 * dt

const FLUID_TAG = 1
const WALL_TAG = 2

@kwdef mutable struct Particle <: AbstractParticle
    # must have properties:
    x_vec_::RealVector = kVec0
    rho_::Float64 = rho_0
    mass_::Float64 = mass
    type_::Int64 = FLUID_TAG
    # additional properties:
    p_::Float64 = 0.0
    drho_::Float64 = 0.0
    v_vec_::RealVector = kVec0
    dv_vec_::RealVector = kVec0
    c_::Float64 = c
    mu_::Float64 = mu
    gap_::Float64 = gap
    sum_kernel_weight_::Float64 = 0.0
    sum_kernel_weighted_value_::Float64 = 0.0
    normal_vec_::RealVector = kVec0
end

@inline function updateDensityAndPressure!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        libUpdateDensity!(p; dt = dt / 2)
        p.p_ = getPressureFromDensity(p.rho_)
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function continuity!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == FLUID_TAG && q.type_ == FLUID_TAG
        kernel_gradient = kernelGradient(r, smooth_kernel)
        libContinuity!(p, q, r; kernel_gradient = kernel_gradient)
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function momentum!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == FLUID_TAG && q.type_ == FLUID_TAG
        # * pressure term
        kernel_value = kernelValue(r, smooth_kernel)
        kernel_gradient = kernelGradient(r, smooth_kernel)
        mean_gap = (p.gap_ + q.gap_) / 2
        reference_kernel_value = kernelValue(mean_gap, smooth_kernel)
        libPressureForce!(
            p,
            q,
            r;
            kernel_value = kernel_value,
            kernel_gradient = kernel_gradient,
            reference_kernel_value = reference_kernel_value,
        )
        # * viscosity term
        libViscosityForce!(p, q, r; kernel_gradient = kernel_gradient, h = h / 2)
        return nothing
    elseif p.type_ == FLUID_TAG && q.type_ == WALL_TAG
        # * viscosity term
        kernel_gradient = kernelGradient(r, smooth_kernel)
        libViscosityForce!(p, q, r; kernel_gradient = kernel_gradient, h = h / 2)
        # * compulsive term
        libCompulsiveForce!(p, q, r; h = h / 2)
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function updateVelocity!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        libUpdateVelocity!(p; dt = dt / 2, body_force_vec = g)
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function updatePosition!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        libUpdatePosition!(p; dt = dt / 2)
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function densityFilter!(p::Particle, q::Particle, r::Float64;)::Nothing
    if p.type_ == FLUID_TAG && q.type_ == FLUID_TAG
        libKernelAverageDensityFilter!(p, q, r; smooth_kernel = smooth_kernel)
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function densityFilter!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        libKernelAverageDensityFilter!(p; smooth_kernel = smooth_kernel)
        p.p_ = getPressureFromDensity(p.rho_)
        return nothing
    else
        return nothing
    end
    return nothing
end

function createRectangleParticles(
    ParticleType::DataType,
    x0::Float64,
    y0::Float64,
    width::Float64,
    height::Float64,
    reference_dr::Float64;
    modifyOnParticle!::Function = EtherSPHCells.noneFunction!,
)::Vector{ParticleType}
    particles = Vector{ParticleType}()
    n_along_x = Int64(width / reference_dr |> round)
    n_along_y = Int64(height / reference_dr |> round)
    dx = width / n_along_x
    dy = height / n_along_y
    for i in 1:n_along_x, j in 1:n_along_y
        particle = ParticleType()
        x = x0 + (i - 0.5) * dx
        y = y0 + (j - 0.5) * dy
        particle.x_vec_ = RealVector(x, y, 0.0)
        modifyOnParticle!(particle)
        particle.mass_ = particle.rho_ * dx * dy
        push!(particles, particle)
    end
    return particles
end

const x0 = 0.0
const y0 = 0.0

function initialPressure!(p::Particle)::Nothing
    depth = water_height - p.x_vec_[2]
    p.p_ = rho_0 * gravity * depth
    p.rho_ = getDensityFromPressure(p.p_)
    return nothing
end

particles = Particle[]

fluid_particles =
    createRectangleParticles(Particle, x0, y0, water_width, water_height, dr; modifyOnParticle! = initialPressure!)
append!(particles, fluid_particles)

@inline function bottomParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = kVecY
    p.type_ = WALL_TAG
    p.mu_ *= 1000
    return nothing
end
bottom_wall_particles = createRectangleParticles(
    Particle,
    x0,
    y0 - wall_width,
    box_width,
    wall_width,
    dr;
    modifyOnParticle! = bottomParticleModify!,
)
append!(particles, bottom_wall_particles)

@inline function leftWallParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = kVecX
    p.type_ = WALL_TAG
    p.mu_ *= 1000
    return nothing
end
left_wall_particles = createRectangleParticles(
    Particle,
    x0 - wall_width,
    y0,
    wall_width,
    box_height,
    dr;
    modifyOnParticle! = leftWallParticleModify!,
)
append!(particles, left_wall_particles)

@inline function rightWallParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = -kVecX
    p.type_ = WALL_TAG
    p.mu_ *= 1000
    return nothing
end
right_wall_particles = createRectangleParticles(
    Particle,
    x0 + box_width,
    y0,
    wall_width,
    box_height,
    dr;
    modifyOnParticle! = rightWallParticleModify!,
)
append!(particles, right_wall_particles)

@inline function topWallParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = -kVecY
    p.type_ = WALL_TAG
    p.mu_ *= 1000
    return nothing
end
top_wall_particles = createRectangleParticles(
    Particle,
    x0,
    y0 + box_height,
    box_width,
    wall_width,
    dr;
    modifyOnParticle! = topWallParticleModify!,
)
append!(particles, top_wall_particles)

@inline function leftBottomCornerParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = (kVecX + kVecY) / sqrt(2)
    p.type_ = WALL_TAG
    p.mu_ *= 1000
    return nothing
end
left_bottom_corner_particles = createRectangleParticles(
    Particle,
    x0 - wall_width,
    y0 - wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = leftBottomCornerParticleModify!,
)
append!(particles, left_bottom_corner_particles)

@inline function rightBottomCornerParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = (-kVecX + kVecY) / sqrt(2)
    p.type_ = WALL_TAG
    p.mu_ *= 1000
    return nothing
end
right_bottom_corner_particles = createRectangleParticles(
    Particle,
    x0 + box_width,
    y0 - wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = rightBottomCornerParticleModify!,
)
append!(particles, right_bottom_corner_particles)

@inline function leftTopCornerParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = (kVecX - kVecY) / sqrt(2)
    p.type_ = WALL_TAG
    p.mu_ *= 1000
    return nothing
end
left_top_corner_particles = createRectangleParticles(
    Particle,
    x0 - wall_width,
    y0 + box_height,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = leftTopCornerParticleModify!,
)
append!(particles, left_top_corner_particles)

@inline function rightTopCornerParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = (-kVecX - kVecY) / sqrt(2)
    p.type_ = WALL_TAG
    p.mu_ *= 1000
    return nothing
end
right_top_corner_particles = createRectangleParticles(
    Particle,
    x0 + box_width,
    y0 + box_height,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = rightTopCornerParticleModify!,
)
append!(particles, right_top_corner_particles)

start_point = RealVector(-h, -h, 0.0)
end_point = RealVector(box_width + h, box_height + h, 0.0)
system = ParticleSystem(Particle, h, start_point, end_point)
append!(system.particles_, particles)

vtp_io = VTPIO()
@inline getPressure(p::Particle)::Float64 = p.p_
@inline getVelocity(p::Particle)::RealVector = p.v_vec_
@inline getNormalVector(p::Particle)::RealVector = p.normal_vec_
addScalar!(vtp_io, "Pressure", getPressure)
addVector!(vtp_io, "Velocity", getVelocity)
addVector!(vtp_io, "Normal", getNormalVector)

vtp_io.step_digit_ = 4
vtp_io.file_name_ = "cruchaga_2d"
vtp_io.output_path_ = "example/results/cruchaga_2d_results"
vtp_io

function main()::Nothing
    assurePathExist(vtp_io)
    t = 0.0
    saveVTP(vtp_io, system, 0, t)
    updateBackgroundCellList!(system)
    applyInteraction!(system, momentum!)
    for step in ProgressBar(1:round(Int, t_end / dt))
        applySelfaction!(system, updateVelocity!)
        applySelfaction!(system, updatePosition!)
        updateBackgroundCellList!(system)
        applyInteraction!(system, continuity!)
        applySelfaction!(system, updateDensityAndPressure!)
        applySelfaction!(system, updatePosition!)
        updateBackgroundCellList!(system)
        applyInteraction!(system, momentum!)
        applySelfaction!(system, updateVelocity!)
        if step % round(Int, output_dt / dt) == 0
            saveVTP(vtp_io, system, step, t)
        end
        if step % round(Int, density_filter_dt / dt) == 0
            applyInteraction!(system, densityFilter!)
            applySelfaction!(system, densityFilter!)
        end
        t += dt
    end
    return nothing
end
