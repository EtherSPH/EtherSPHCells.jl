#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/22 21:13:10
  @ license: MIT
  @ description:
 =#

using EtherSPHCells
using Parameters

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
const nu = mu / rho_0

const dt = 0.1 * h / c
const t_end = 3.0
const output_dt = 100 * dt
const density_filter_dt = 30 * dt

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

@inline function updateDensityAndPressure!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        p.rho_ += p.drho_ * dt / 2
        p.drho_ = 0.0
        p.p_ = c^2 * (p.rho_ - rho_0)
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function continuity!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == FLUID_TAG && q.type_ == FLUID_TAG
        kernel_gradient = kernelGradient(r, smooth_kernel)
        p.drho_ += q.mass_ * dot(p.v_vec_ - q.v_vec_, p.x_vec_ - q.x_vec_) * kernel_gradient / r
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
        p_rho_2 = p.p_ / p.rho_^2 + q.p_ / q.rho_^2
        reference_kernel_value = kernelValue(mean_gap, smooth_kernel)
        p_rho_2 += abs(p_rho_2) * 0.01 * kernel_value / reference_kernel_value
        p.dv_vec_ += -q.mass_ * p_rho_2 * kernel_gradient / r * (p.x_vec_ - q.x_vec_)
        # * viscosity term
        mean_mu = 2 * p.mu_ * q.mu_ / (p.mu_ + q.mu_)
        sum_rho = p.rho_ + q.rho_
        viscosity_force = 8 * mean_mu * kernel_gradient * r / sum_rho^2 / (r^2 + 0.01 * h^2)
        p.dv_vec_ += q.mass_ * viscosity_force * (q.v_vec_ - p.v_vec_)
        return nothing
    elseif p.type_ == FLUID_TAG && q.type_ == WALL_TAG
        # * viscosity term
        kernel_gradient = kernelGradient(r, smooth_kernel)
        mean_mu = 2 * p.mu_ * q.mu_ / (p.mu_ + q.mu_)
        sum_rho = p.rho_ + q.rho_
        viscosity_force = 8 * mean_mu * kernel_gradient * r / sum_rho^2 / (r^2 + 0.01 * h^2)
        p.dv_vec_ += q.mass_ * viscosity_force * (q.v_vec_ - p.v_vec_)
        # * compulsive term
        psi = abs(dot(p.x_vec_ - q.x_vec_, q.normal_vec_))
        xi = sqrt(max(0.0, r^2 - psi^2))
        eta = psi / q.gap_
        if eta > 1 || xi > q.gap_
            return nothing
        end
        p_xi = abs(1 + cos(pi * xi / q.gap_) / 2)
        verticle_v = dot(p.v_vec_ - q.v_vec_, q.normal_vec_)
        beta = verticle_v > 0 ? 0.0 : 1.0
        r_psi = (0.01 * p.c_^2 + beta * p.c_ * abs(verticle_v)) / h * abs(1 - eta) / sqrt(eta)
        wall_force = r_psi * p_xi
        p.dv_vec_ += wall_force * q.normal_vec_
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function updateVelocity!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        p.v_vec_ += (p.dv_vec_ + g) * dt / 2
        p.dv_vec_ = kVec0
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function updatePosition!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        p.x_vec_ += p.v_vec_ * dt
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function densityFilter!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == FLUID_TAG && q.type_ == FLUID_TAG
        kernel_value = kernelValue(r, smooth_kernel)
        p.sum_kernel_weighted_value_ += q.mass_ * kernel_value
        p.sum_kernel_weight_ += q.mass_ / q.rho_ * kernel_value
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function densityFilter!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        p.sum_kernel_weighted_value_ += p.mass_ * smooth_kernel.kernel_0_
        p.sum_kernel_weight_ += p.mass_ / p.rho_ * smooth_kernel.kernel_0_
        p.rho_ = p.sum_kernel_weighted_value_ / p.sum_kernel_weight_
        p.p_ = c^2 * (p.rho_ - rho_0)
        p.sum_kernel_weight_ = 0.0
        p.sum_kernel_weighted_value_ = 0.0
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
    modifyOnParticle!::Function = EtherSPHCells.selfaction!,
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
    p.rho_ += p.p_ / p.c_^2
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
