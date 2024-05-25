#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/24 21:45:34
  @ license: MIT
  @ description:
 =#

using EtherSPHCells
using Parameters
using ProgressBars

const prandtl_number = 0.71
const rayleigh_number = 1e3

const dim = 2
const dr = 0.01
const h = 3 * dr
const wall_width = h

const smooth_kernel = SmoothKernel(h, dim, CubicSpline)

const cavity_length = 1.0
const rho_0 = 1.0
const mass = rho_0 * dr^dim
const c_0 = 3.0
const nu_0 = prandtl_number / sqrt(rayleigh_number)
const mu_0 = rho_0 * nu_0
const mu_wall = mu_0
const p_0 = 0.02 * rho_0 * c_0^2
const alpha = nu_0 / prandtl_number

const gravity = 1.0
const g = RealVector(0.0, -gravity, 0.0)
const t_left = 1.0
const t_right = 0.0
const delta_t = t_left - t_right
const mean_delta_t = 0.5 * (t_left + t_right)
const t_0 = mean_delta_t
const kappa = 1.0
const cp = prandtl_number * kappa / mu_0
const beta = rayleigh_number * nu_0 * alpha / gravity / delta_t / cavity_length^3
const gamma = 7

const dt = 0.1 * h / c_0
const t_end = 30.0
const output_dt = 100 * dt
const density_filter_dt = 5 * dt

const FLUID_TAG = 1
const WALL_TAG = 2
const THERMOSTATIC_WALL_TAG = 3

@inline function getPressureFromDensity(rho::Float64)::Float64
    ratio = rho / rho_0
    return c_0^2 / gamma * (ratio^gamma - 1) * rho_0 + p_0
end

@inline function bodyForceVectorByBoussinesqApproximation(t::Float64)::RealVector
    return -g * beta * (t - t_0)
end

@kwdef mutable struct Particle <: AbstractParticle
    # must have properties:
    x_vec_::RealVector = kVec0
    rho_::Float64 = rho_0
    mass_::Float64 = mass
    type_::Int64 = FLUID_TAG
    # additional properties:
    # * first group, flow field properties
    p_::Float64 = p_0
    drho_::Float64 = 0.0
    v_vec_::RealVector = kVec0
    dv_vec_::RealVector = kVec0
    c_::Float64 = c_0
    mu_::Float64 = mu_0
    gap_::Float64 = dr
    sum_kernel_weight_::Float64 = 0.0
    sum_kernel_weighted_value_::Float64 = 0.0
    # * second group, wall properties
    normal_vec_::RealVector = kVec0
    # * third group, temperature field properties
    # ρcₚ∂T/∂t = ∇·(k∇T)
    t_::Float64 = t_0
    dt_::Float64 = 0.0
    kappa_::Float64 = kappa
    cp_::Float64 = cp
end

@inline function updateDensityAndPressure!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        libUpdateDensity!(p; dt = dt)
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
        # ! thermal conduction term can be added here together in momentum equation
        libThermalConduction!(p, q, r; kernel_gradient = kernel_gradient, h = h / 2)
        return nothing
    elseif p.type_ == FLUID_TAG && q.type_ == WALL_TAG
        kernel_value = kernelValue(r, smooth_kernel)
        kernel_gradient = kernelGradient(r, smooth_kernel)
        mean_gap = (p.gap_ + q.gap_) / 2
        reference_kernel_value = kernelValue(mean_gap, smooth_kernel)
        # * pressure term
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
        # * compulsive term
        libCompulsiveForce!(p, q, r; h = h / 2)
        return nothing
    elseif p.type_ == FLUID_TAG && q.type_ == THERMOSTATIC_WALL_TAG
        kernel_value = kernelValue(r, smooth_kernel)
        kernel_gradient = kernelGradient(r, smooth_kernel)
        mean_gap = (p.gap_ + q.gap_) / 2
        reference_kernel_value = kernelValue(mean_gap, smooth_kernel)
        # * pressure term
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
        # * compulsive term
        libCompulsiveForce!(p, q, r; h = h / 2)
        # ! thermal conduction term can be added here together in momentum equation
        libThermalConduction!(p, q, r; kernel_gradient = kernel_gradient, h = h / 2)
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function updateVelocityAndPosition!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        # libUpdateVelocityAndPosition!(p; dt = dt)
        # ! thermal conduction term can be added here together in momentum equation
        libUpdateTemperature!(p; dt = dt)
        body_force = bodyForceVectorByBoussinesqApproximation(p.t_)
        libUpdateVelocityAndPosition!(p; dt = dt, body_force_vec = body_force)
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

particles = Particle[]

fluid_particles = createRectangleParticles(Particle, x0, y0, cavity_length, cavity_length, dr)
append!(particles, fluid_particles)

@inline function modifyOnleftWall!(p::Particle)::Nothing
    p.type_ = THERMOSTATIC_WALL_TAG
    p.normal_vec_ = kVecX
    p.t_ = t_left
    p.mu_ = mu_wall
    return nothing
end
left_wall_particles = createRectangleParticles(
    Particle,
    x0 - wall_width,
    y0 - wall_width,
    wall_width,
    cavity_length + 2 * wall_width,
    dr;
    modifyOnParticle! = modifyOnleftWall!,
)
append!(particles, left_wall_particles)

@inline function modifyOnrightWall!(p::Particle)::Nothing
    p.type_ = THERMOSTATIC_WALL_TAG
    p.normal_vec_ = -kVecX
    p.t_ = t_right
    p.mu_ = mu_wall
    return nothing
end
right_wall_particles = createRectangleParticles(
    Particle,
    x0 + cavity_length,
    y0 - wall_width,
    wall_width,
    cavity_length + 2 * wall_width,
    dr;
    modifyOnParticle! = modifyOnrightWall!,
)
append!(particles, right_wall_particles)

@inline function modifyOnbottomWall!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = kVecY
    p.mu_ = mu_wall
    p.t_ = NaN
    return nothing
end
bottom_wall_particles = createRectangleParticles(
    Particle,
    x0,
    y0 - wall_width,
    cavity_length,
    wall_width,
    dr;
    modifyOnParticle! = modifyOnbottomWall!,
)
append!(particles, bottom_wall_particles)

@inline function modifyOntopWall!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = -kVecY
    p.mu_ = mu_wall
    p.t_ = NaN
    return nothing
end
top_wall_particles = createRectangleParticles(
    Particle,
    x0,
    y0 + cavity_length,
    cavity_length,
    wall_width,
    dr;
    modifyOnParticle! = modifyOntopWall!,
)
append!(particles, top_wall_particles)

start_point = RealVector(-h, -h, 0.0)
end_point = RealVector(cavity_length + h, cavity_length + h, 0.0)
system = ParticleSystem(Particle, h, start_point, end_point)
append!(system.particles_, particles)

vtp_io = VTPIO()
@inline getPressure(p::Particle)::Float64 = p.p_
@inline getVelocity(p::Particle)::RealVector = p.v_vec_
@inline getNormalVector(p::Particle)::RealVector = p.normal_vec_
@inline getTemperature(p::Particle)::Float64 = p.t_
addScalar!(vtp_io, "Pressure", getPressure)
addVector!(vtp_io, "Velocity", getVelocity)
addVector!(vtp_io, "Normal", getNormalVector)
addScalar!(vtp_io, "Temperature", getTemperature)

vtp_io.step_digit_ = 4
vtp_io.file_name_ = "natural_convection_cavity_ra_1e3_"
vtp_io.output_path_ = "example/results/natural_convection_cavity/ra_1e3"
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
