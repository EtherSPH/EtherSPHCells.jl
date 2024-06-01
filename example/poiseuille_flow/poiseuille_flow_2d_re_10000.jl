#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/28 16:03:10
  @ license: MIT
  @ description:
 =#

using EtherSPHCells
using Parameters
using ProgressBars

const dim = 2
const dr = 1e-2 / 50
const h = 3.0 * dr

const smooth_kernel = SmoothKernel(h, dim, WendlandC2)

const x0 = 0.0
const y0 = 0.0
const pipe_width = 1e-2
const pipe_length = 6e-2
const buffer_length = 6 * dr
const wall_thickness = h
const wall_length = pipe_length + 2 * buffer_length

const start_point = RealVector(x0 - buffer_length, y0 - wall_thickness, 0.0)
const end_point = RealVector(x0 + pipe_length + buffer_length, y0 + pipe_width + wall_thickness, 0.0)

# νd²u/dy² = 1/ρ dp/dx, assuming -1/ρ dp/dx = f = const, which acts like a body force
# apply non-slip boundary condition at y=0 and y=l
# u = ky(l-y), d²u/dy² = -2k = -f / ν, k = f/2ν
# u = f/2ν * y(l-y)
# uₘₐₓ = f/8ν * l²
# ̄u = f/12ν * l²
const f = 8e-2
const g = RealVector(f, 0.0, 0.0)
const rho_0 = 1e3
const mass = rho_0 * dr^dim
const mu = 1e-3
const mu_solid = mu * 1e3
const nu = mu / rho_0
const l = pipe_width

@inline function getVelocityFromY(y::Float64)
    return f / 2 / nu * y * (l - y)
end

const u_max = getVelocityFromY(l / 2)
const c = u_max * 10
const u_mean = u_max / 3 * 2
const p_0 = c^2 * 0.02 * rho_0
const gamma = 7

@inline function getPressureFromDensity(rho::Float64)::Float64
    ratio = rho / rho_0
    return c^2 / gamma * (ratio^gamma - 1) * rho_0 + p_0
end

@info "Re ≈ $(rho_0 * u_max * l / mu)"

const FLUID_TAG = 1
const WALL_TAG = 2
const INFLOW_TAG = 3
const OUTFLOW_TAG = 4

const dt = 0.1 * h / c
const t_end = 0.3
const output_dt = 100 * dt
const density_filter_dt = 10 * dt

@kwdef mutable struct Particle <: AbstractParticle
    # must have properties:
    x_vec_::RealVector = kVec0
    rho_::Float64 = rho_0
    mass_::Float64 = mass
    type_::Int64 = FLUID_TAG
    # additional properties:
    p_::Float64 = p_0
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

@inline function setTheoreticalVelocity!(p::Particle)::Nothing
    p.v_vec_ = kVecX * getVelocityFromY(p.x_vec_[2])
    return nothing
end

@inline function updateDensityAndPressure!(p::Particle)::Nothing
    if p.type_ == INFLOW_TAG
        return nothing
    else
        libUpdateDensity!(p; dt = dt)
        p.p_ = getPressureFromDensity(p.rho_)
        return nothing
    end
    return nothing
end

@inline function continuity!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == INFLOW_TAG
        return nothing
    else
        kernel_gradient = kernelGradient(r, smooth_kernel)
        libContinuity!(p, q, r; kernel_gradient = kernel_gradient)
        return nothing
    end
    return nothing
end

@inline function momentum!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == FLUID_TAG && q.type_ in (FLUID_TAG, WALL_TAG, OUTFLOW_TAG)
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
    elseif p.type_ == FLUID_TAG && q.type_ == INFLOW_TAG
        # * pressure term
        kernel_value = kernelValue(r, smooth_kernel)
        kernel_gradient = kernelGradient(r, smooth_kernel)
        mean_gap = (p.gap_ + q.gap_) / 2
        reference_kernel_value = kernelValue(mean_gap, smooth_kernel)
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
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function updateVelocityAndPosition!(p::Particle)::Nothing
    if p.type_ in (FLUID_TAG, OUTFLOW_TAG)
        libUpdateVelocity!(p; dt = dt, body_force_vec = g)
        libUpdatePosition!(p; dt = dt)
    elseif p.type_ == INFLOW_TAG
        libUpdatePosition!(p; dt = dt)
    else
        nothing
    end
    p.dv_vec_ = kVec0
    return nothing
end

@inline function densityFilter!(p::Particle, q::Particle, r::Float64;)::Nothing
    if p.type_ == FLUID_TAG
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

@kwdef mutable struct ThreadSafeParticleVector
    particles_::Vector{Particle} = Particle[]
    thread_safe_lock_::Base.Threads.ReentrantLock = Base.Threads.ReentrantLock()
end

@inline function appendParticle!(thspv::ThreadSafeParticleVector, p::Particle)::Nothing
    try
        lock(thspv.thread_safe_lock_)
        push!(thspv.particles_, p)
    finally
        unlock(thspv.thread_safe_lock_)
    end
    return nothing
end
@inline function clear!(thspv::ThreadSafeParticleVector)::Nothing
    empty!(thspv.particles_)
    return nothing
end
@inline function appendParticles!(system::ParticleSystem, thspv::ThreadSafeParticleVector)::Nothing
    append!(system.particles_, thspv.particles_)
    return nothing
end
@inline function addNewParticles!(system::ParticleSystem, thspv::ThreadSafeParticleVector)::Nothing
    clear!(thspv)
    Threads.@threads for i in eachindex(system.particles_)
        p = system.particles_[i]
        if p.type_ == INFLOW_TAG && p.x_vec_[1] >= x0
            p_copy = deepcopy(p)
            p_copy.x_vec_ -= kVecX * buffer_length
            p.type_ = FLUID_TAG
            setTheoreticalVelocity!(p_copy)
            appendParticle!(thspv, p_copy)
        elseif p.type_ == FLUID_TAG && p.x_vec_[1] >= x0 + pipe_length
            p.type_ = OUTFLOW_TAG
            setTheoreticalVelocity!(p)
            p.rho_ = rho_0
            p.p_ = p_0
        end
    end
    appendParticles!(system, thspv)
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
        particle.mass_ = particle.rho_ * dx * dy
        if modifyOnParticle!(particle) == true
            push!(particles, particle)
        end
    end
    return particles
end

particles = Particle[]

@inline function initialFluidParticle!(p::Particle)::Bool
    p.type_ = FLUID_TAG
    setTheoreticalVelocity!(p)
    return true
end
fluid_particles =
    createRectangleParticles(Particle, x0, y0, pipe_length, pipe_width, dr; modifyOnParticle! = initialFluidParticle!)
append!(particles, fluid_particles)

@inline function initialBottomWallParticle!(p::Particle)::Bool
    p.type_ = WALL_TAG
    p.normal_vec_ = kVecY
    p.mu_ = mu_solid
    return true
end
bottom_wall_particles = createRectangleParticles(
    Particle,
    x0 - buffer_length,
    y0 - wall_thickness,
    wall_length,
    wall_thickness,
    dr;
    modifyOnParticle! = initialBottomWallParticle!,
)
append!(particles, bottom_wall_particles)

@inline function initialTopWallParticle!(p::Particle)::Bool
    p.type_ = WALL_TAG
    p.normal_vec_ = -kVecY
    p.mu_ = mu_solid
    return true
end
top_wall_particles = createRectangleParticles(
    Particle,
    x0 - buffer_length,
    y0 + pipe_width,
    wall_length,
    wall_thickness,
    dr;
    modifyOnParticle! = initialTopWallParticle!,
)
append!(particles, top_wall_particles)

@inline function initialInflowParticle!(p::Particle)::Bool
    p.type_ = INFLOW_TAG
    setTheoreticalVelocity!(p)
    return true
end
inflow_particles = createRectangleParticles(
    Particle,
    x0 - buffer_length,
    y0,
    buffer_length,
    pipe_width,
    dr;
    modifyOnParticle! = initialInflowParticle!,
)
append!(particles, inflow_particles)

@inline function initialOutflowParticle!(p::Particle)::Bool
    p.type_ = OUTFLOW_TAG
    setTheoreticalVelocity!(p)
    return true
end
outflow_particles = createRectangleParticles(
    Particle,
    x0 + pipe_length,
    y0,
    buffer_length,
    pipe_width,
    dr;
    modifyOnParticle! = initialOutflowParticle!,
)
append!(particles, outflow_particles)

system = ParticleSystem(Particle, h, start_point, end_point)
append!(system.particles_, particles)
thapv = ThreadSafeParticleVector()

vtp_io = VTPIO()
@inline getPressure(p::Particle)::Float64 = p.p_
@inline getVelocity(p::Particle)::RealVector = p.v_vec_
@inline getNormalVector(p::Particle)::RealVector = p.normal_vec_
addScalar!(vtp_io, "Pressure", getPressure)
addVector!(vtp_io, "Velocity", getVelocity)
addVector!(vtp_io, "Normal", getNormalVector)

vtp_io.step_digit_ = 4
vtp_io.file_name_ = "poiseuille_flow_2d_re_10000_"
vtp_io.output_path_ = "example/results/poiseuille_flow/2d_re_10000"
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
        addNewParticles!(system, thapv)
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
