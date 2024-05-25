#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/25 13:37:39
  @ license: MIT
  @ description:
 =#

using EtherSPHCells
using Parameters
using ProgressBars

# case 1: ratio_io = 3., c_0 = 2.0, p_0 = 0.025*..., rayleigh_number = 1e3/4/5

const prandtl_number = 0.706
const rayleigh_number = 1e3

const r_outer = 1.0
const ratio_io = 3.0
const r_inner = r_outer / ratio_io
const reference_length = r_outer - r_inner

const dim = 2
const dr = 0.01
const h = 3 * dr
const wall_width = h

const smooth_kernel = SmoothKernel(h, dim, WendlandC2)

const gravity = 1.0
const g = RealVector(0.0, -gravity, 0.0)
const beta = 0.05
const mu_0 = 1e-3
const mu_wall = mu_0
const kappa = 1.0

const t_outer = 0.0
const t_inner = 1.0
const delta_t = t_inner - t_outer
const t_0 = t_outer

const cp = prandtl_number * kappa / mu_0
const rho_0 = sqrt(rayleigh_number * mu_0 * kappa / gravity / beta / (reference_length^3) / delta_t / cp)
const mass = rho_0 * dr^dim
const c_0 = 2.0
const p_0 = 0.04 * rho_0 * c_0^2
const gamma = 7
const alpha = mu_0 / rho_0 / prandtl_number

const dt = 0.1 * h / c_0
const t_end = 200.0
const output_dt = 100 * dt
const density_filter_dt = 5 * dt

const FLUID_TAG = 1
const THERMOSTATIC_WALL_TAG = 2

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

function createRingParticles(
    ParticleType::DataType,
    x_c::Float64,
    y_c::Float64,
    r_in::Float64,
    r_out::Float64,
    reference_dr::Float64;
    modifyOnParticle!::Function = EtherSPHCells.noneFunction!,
)::Vector{Particle}
    particles = Vector{Particle}()
    n_r_layer = Int64(round((r_out - r_in) / reference_dr))
    dr_r = (r_out - r_in) / n_r_layer
    for n_r in 1:n_r_layer
        r = r_in + (n_r - 0.5) * dr_r
        n_theta_layer = Int64(round(2 * pi * r / reference_dr))
        dr_theta = 2 * pi * r / n_theta_layer
        for n_theta in 1:n_theta_layer
            particle = ParticleType()
            theta = n_theta * 2 * pi / n_theta_layer
            x = x_c + r * cos(theta)
            y = y_c + r * sin(theta)
            particle.x_vec_ = RealVector(x, y, 0.0)
            particle.mass_ = particle.rho_ * dr_r * r * dr_theta
            modifyOnParticle!(particle)
            push!(particles, particle)
        end
    end
    return particles
end

const x0 = 0.0
const y0 = 0.0

particles = Particle[]

fluid_particles = createRingParticles(Particle, x0, y0, r_inner, r_outer, dr;)
append!(particles, fluid_particles)

@inline function modifyInnerWallParticle!(p::Particle)::Nothing
    p.type_ = THERMOSTATIC_WALL_TAG
    p.mu_ = mu_wall
    p.normal_vec_ = normalize(p.x_vec_ - RealVector(x0, y0, 0.0))
    p.t_ = t_inner
    return nothing
end
inner_wall_particles = createRingParticles(
    Particle,
    x0,
    y0,
    r_inner - wall_width,
    r_inner,
    dr;
    modifyOnParticle! = modifyInnerWallParticle!,
)
append!(particles, inner_wall_particles)

@inline function modifyOuterWallParticle!(p::Particle)::Nothing
    p.type_ = THERMOSTATIC_WALL_TAG
    p.mu_ = mu_wall
    p.normal_vec_ = normalize(RealVector(x0, y0, 0.0) - p.x_vec_)
    p.t_ = t_outer
    return nothing
end

outer_wall_particles = createRingParticles(
    Particle,
    x0,
    y0,
    r_outer,
    r_outer + wall_width,
    dr;
    modifyOnParticle! = modifyOuterWallParticle!,
)
append!(particles, outer_wall_particles)

start_point = RealVector(x0 - r_outer - wall_width, y0 - r_outer - wall_width, 0.0)
end_point = RealVector(x0 + r_outer + wall_width, y0 + r_outer + wall_width, 0.0)
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
vtp_io.file_name_ = "natural_convection_circular_center_3.0_ra_1e3_"
vtp_io.output_path_ = "example/results/natural_convection_circular/center_3.0_ra_1e3"
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
