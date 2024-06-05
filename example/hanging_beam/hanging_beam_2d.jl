#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/06/04 20:39:38
  @ license: MIT
  @ description:
 =#

using EtherSPHCells
using Parameters
using ProgressBars

const dim = 2
const beam_length = 0.2
const beam_width = 0.02
const beam_indside_length = 0.06
const dr = beam_width / 20
const h = 2.6 * dr
const beam_wall_thickness = 3 * dr
const gap = dr
# Zhang, - 2023 - Essentially non-hourglass and non-tensile-instabil.pdf, arXiv
const zeta = 0.7 * dim + 3.5 # ζ, a magic number which is used to control the numerical stability, proposed by Zhang et al.

const smooth_kernel = SmoothKernel(h, dim, CubicSpline)

const rho_0 = 1000.0
const mass = rho_0 * h^dim
const young_modulus = 2e6 # E
const poisson_ratio = 0.3975 # ν
const shear_modulus = young_modulus / 2 / (1 + poisson_ratio) # G
const bulk_modulus = young_modulus / 3 / (1 - 2 * poisson_ratio) # K
const c_0 = sqrt(bulk_modulus / rho_0)

const l = beam_length
const k = 1.875 / l
const kl = k * l
const hh = beam_width
const omega = sqrt(young_modulus * hh^2 * k^4 / 12 / rho_0 / (1 - poisson_ratio^2))
@info "T = $(2 * pi / omega)"

@inline function f(x::Float64)::Float64
    return (sin(kl) + sinh(kl)) * (cos(k * x) - cosh(k * x)) - (cos(kl) + cosh(kl)) * (sin(k * x) - sinh(k * x))
end

@inline function verticalVelocity(x::Float64)::RealVector
    if x < 0.0
        return kVec0
    else
        return kVecY * c_0 * f(x) / f(l) * 0.01
    end
end

const MATERIAL_MOVABLE_TAG = 1
const MATERIAL_FIXED_TAG = 2

const dt = 0.1 * h / c_0
const t_end = 1.5
const output_dt = 500 * dt
@info "total steps: $(round(Int, t_end / dt))"

@kwdef mutable struct Particle <: AbstractParticle
    # must have properties:
    x_vec_::RealVector = kVec0
    rho_::Float64 = rho_0
    mass_::Float64 = mass
    type_::Int64 = MATERIAL_MOVABLE_TAG
    # additional properties:
    p_::Float64 = 0.0
    drho_::Float64 = 0.0
    v_vec_::RealVector = kVec0
    dv_vec_::RealVector = kVec0
    dv_shear_vec_::RealVector = kVec0
    gap_::Float64 = gap
end

@inline function getPressureFromDensity(rho::Float64)::Float64
    return c_0^2 * (rho - rho_0)
end

@inline function continuityAndShearAcceleration!(p::Particle, q::Particle, r::Float64)::Nothing
    kernel_gradient = kernelGradient(r, smooth_kernel)
    libContinuity!(p, q, r; kernel_gradient = kernel_gradient)
    p.dv_shear_vec_ +=
        2 * zeta * shear_modulus / p.rho_ * dt * q.mass_ / q.rho_ * kernel_gradient * (p.x_vec_ - q.x_vec_) / r *
        dot(p.v_vec_ - q.v_vec_, p.x_vec_ - q.x_vec_) / r^2
    return nothing
end

@inline function updateDensityAndPressure!(p::Particle)::Nothing
    libUpdateDensity!(p)
    p.p_ = getPressureFromDensity(p.rho_)
    return nothing
end

@inline function momentum!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == MATERIAL_MOVABLE_TAG
        kernel_gradient = kernelGradient(r, smooth_kernel)
        kernel_value = kernelValue(r, smooth_kernel)
        mean_gap = (p.gap_ + q.gap_) / 2
        reference_kernel_value = kernelValue(mean_gap, smooth_kernel)
        p_rho_2 = (p.p_ + q.p_) / p.rho_ / q.rho_
        p_rho_2 += 0.01 * abs(p_rho_2) * kernel_value / reference_kernel_value
        p.dv_vec_ += -q.mass_ * p_rho_2 * kernel_gradient * (p.x_vec_ - q.x_vec_) / r
    end
    return nothing
end

@inline function updateVelocityAndPosition!(p::Particle)::Nothing
    if p.type_ == MATERIAL_MOVABLE_TAG
        p.dv_vec_ += p.dv_shear_vec_
        libUpdateVelocityAndPosition!(p; dt = dt)
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

@inline function initializeBeam!(p::Particle)::Nothing
    p.v_vec_ = verticalVelocity(p.x_vec_[1] - x0)
    p.type_ = MATERIAL_MOVABLE_TAG
    return nothing
end

@inline function initializeWall!(p::Particle)::Nothing
    p.type_ = MATERIAL_FIXED_TAG
    return nothing
end

const x0 = 0.0
const y0 = 0.0

particles = Particle[]

beam_particles = createRectangleParticles(
    Particle,
    x0 - beam_indside_length,
    y0,
    beam_indside_length + beam_length,
    beam_width,
    dr;
    modifyOnParticle! = initializeBeam!,
)
append!(particles, beam_particles)

bottom_wall_particles = createRectangleParticles(
    Particle,
    x0 - beam_indside_length,
    y0 - beam_wall_thickness,
    beam_indside_length,
    beam_wall_thickness,
    dr;
    modifyOnParticle! = initializeWall!,
)
append!(particles, bottom_wall_particles)

top_wall_particles = createRectangleParticles(
    Particle,
    x0 - beam_indside_length,
    y0 + beam_width,
    beam_indside_length,
    beam_wall_thickness,
    dr;
    modifyOnParticle! = initializeWall!,
)
append!(particles, top_wall_particles)

start_point = RealVector(-beam_indside_length, -beam_wall_thickness - beam_indside_length, 0.0)
end_point = RealVector(beam_length + beam_indside_length, beam_width + beam_indside_length, 0.0)
system = ParticleSystem(Particle, h, start_point, end_point)
append!(system.particles_, particles)

vtp_io = VTPIO()
@inline getPressure(p::Particle)::Float64 = p.p_
@inline getVelocity(p::Particle)::RealVector = p.v_vec_
addScalar!(vtp_io, "Pressure", getPressure)
addVector!(vtp_io, "Velocity", getVelocity)

vtp_io.step_digit_ = 4
vtp_io.file_name_ = "hanging_beam_2d_"
vtp_io.output_path_ = "example/results/hanging_beam/2d"
vtp_io

function main()::Nothing
    assurePathExist(vtp_io)
    t = 0.0
    saveVTP(vtp_io, system, 0, t)
    updateBackgroundCellList!(system)
    # simply use Euler forward method
    for step in ProgressBar(1:round(Int, t_end / dt))
        applyInteraction!(system, continuityAndShearAcceleration!)
        applySelfaction!(system, updateDensityAndPressure!)
        applyInteraction!(system, momentum!)
        applySelfaction!(system, updateVelocityAndPosition!)
        updateBackgroundCellList!(system)
        if step % round(Int, output_dt / dt) == 0
            saveVTP(vtp_io, system, step, t)
        end
        t += dt
    end
    return nothing
end
