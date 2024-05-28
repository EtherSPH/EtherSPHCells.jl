#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/26 21:15:57
  @ license: MIT
  @ description:
 =#

using EtherSPHCells
using Parameters
using ProgressBars

# compared with collapse_dry_compulsive.jl
# an additional wall pressure term is extrapolated from fluid as a mirror pressure term

const dim = 2
const dr = 0.01
const h = 3 * dr

const smooth_kernel = SmoothKernel(h, dim, CubicSpline)

const water_width = 1.0
const water_height = 2.0
const box_width = 4.0
const box_height = 3.0
const wall_width = h

const rho_0 = 1000.0
const mass = rho_0 * dr^dim
const gravity = 9.8
const c = 10 * sqrt(2 * gravity * water_height)
const g = RealVector(0.0, -gravity, 0.0)
const mu = 1e-3
const mu_wall = mu * 1000
const nu = mu / rho_0

const dt = 0.1 * h / c
const t_end = 4.0
const output_dt = 100 * dt
const density_filter_dt = 5 * dt

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
    gap_::Float64 = dr
    sum_kernel_weight_::Float64 = 0.0
    sum_kernel_weighted_value_::Float64 = 0.0
    normal_vec_::RealVector = kVec0
end

@inline function continuity!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == WALL_TAG
        return nothing
    else
        kernel_gradient = kernelGradient(r, smooth_kernel)
        libContinuity!(p, q, r; kernel_gradient = kernel_gradient)
    end
    return nothing
end

@inline function updateDensityAndPressure!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        libUpdateDensity!(p; dt = dt)
        p.p_ = c^2 * (p.rho_ - rho_0)
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
    elseif p.type_ == FLUID_TAG && q.type_ == WALL_TAG
        kernel_value = kernelValue(r, smooth_kernel)
        kernel_gradient = kernelGradient(r, smooth_kernel)
        kernel_value = kernelValue(r, smooth_kernel)
        kernel_gradient = kernelGradient(r, smooth_kernel)
        mean_gap = (p.gap_ + q.gap_) / 2
        reference_kernel_value = kernelValue(mean_gap, smooth_kernel)
        # ! * mirror pressure term
        libMirrorPressureForce!(
            p,
            q,
            r;
            kernel_value = kernel_value,
            kernel_gradient = kernel_gradient,
            reference_kernel_value = reference_kernel_value,
        )
        # * viscosity term
        libViscosityForce!(p, q, r; kernel_gradient = kernel_gradient, h = h / 2)
        # * compulsive term
        libCompulsiveForce!(p, q, r; h = h / 2)
    end
    return nothing
end

@inline function updateVelocityAndPosition!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        libUpdateVelocityAndPosition!(p; dt = dt, body_force_vec = g)
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
        p.p_ = c^2 * (p.rho_ - rho_0)
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

particles = Particle[]

fluid_particles = createRectangleParticles(Particle, x0, y0, water_width, water_height, dr;)
append!(particles, fluid_particles)

@inline function bottomParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = kVecY
    p.type_ = WALL_TAG
    p.mu_ = mu_wall
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
    p.mu_ = mu_wall
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
    p.mu_ = mu_wall
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

@inline function leftBottomCornerParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = (kVecX + kVecY) / sqrt(2)
    p.type_ = WALL_TAG
    p.mu_ = mu_wall
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
    p.mu_ = mu_wall
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
vtp_io.file_name_ = "collapse_dry_mirror"
vtp_io.output_path_ = "example/results/collapse_dry/mirror_wall"
vtp_io

function main()::Nothing
    assurePathExist(vtp_io)
    t = 0.0
    saveVTP(vtp_io, system, 0, t)
    updateBackgroundCellList!(system)
    # simply use Euler forward method
    for step in ProgressBar(1:round(Int, t_end / dt))
        applyInteraction!(system, continuity!)
        applySelfaction!(system, updateDensityAndPressure!)
        applyInteraction!(system, momentum!)
        applySelfaction!(system, updateVelocityAndPosition!)
        updateBackgroundCellList!(system)
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
