#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/22 21:13:10
  @ license: MIT
  @ description:
 =#

# this file is a demo / tutorial for users to learn how to use EtherSPHCells.jl

using EtherSPHCells # import the main module
using Parameters # import the Parameters module, which allows us to define keyword arguments

const dim = 2 # the dimension of the simulation
const dr = 0.01 # the particle spacing, also the size of square particles
const h = 3 * dr # the smoothing length, which is 3 times the particle spacing

const smooth_kernel = SmoothKernel(h, dim, CubicSpline) # choose `CubicSpline` as the smoothing kernel

const water_width = 1.0 # the width of the water
const water_height = 2.0 # the height of the water
const box_width = 4.0 # the width of the box
const box_height = 3.0 # the height of the box
const wall_width = h # the width of the wall, which is the same as the smoothing length

const rho_0 = 1000.0 # the reference density
const mass = rho_0 * dr^dim # the mass of the particle
const gravity = 9.8 # the gravity acceleration
const c = 10 * sqrt(2 * gravity * water_height) # the speed of sound is 10 times of max speed in the water
const g = RealVector(0.0, -gravity, 0.0) # the gravity vector
const mu = 1e-3 # the viscosity coefficient
const nu = mu / rho_0 # the kinematic viscosity coefficient

const dt = 0.1 * h / c # simulation time interval
const t_end = 4.0 # total simulation time
const output_dt = 100 * dt # output time interval
const density_filter_dt = 100 * dt # density filter time interval, recommended as 30 by DualSPHysics, I simply set it as 100 here

const FLUID_TAG = 1 # the tag of fluid particles
const WALL_TAG = 2 # the tag of wall particles

# particle definition rule:
# - `must have properties` are needed in the particle definition.
# - `additional properties` are optional in the particle definition.
# don't worry abount memory allocation, try as best to let particle itself has the property
# instead of global variables.
@kwdef mutable struct Particle <: AbstractParticle
    # must have properties:
    x_vec_::RealVector = kVec0 # the position of the particle
    rho_::Float64 = rho_0 # the density of the particle
    mass_::Float64 = mass # the mass of the particle
    type_::Int64 = FLUID_TAG # the type of the particle, default is fluid
    # additional properties:
    p_::Float64 = 0.0 # the pressure of the particle
    drho_::Float64 = 0.0 # the density change rate of the particle
    v_vec_::RealVector = kVec0 # the velocity of the particle
    dv_vec_::RealVector = kVec0 # the acceleration of the particle
    c_::Float64 = c # the speed of sound of the particle
    mu_::Float64 = mu # the viscosity coefficient of the particle
    gap_::Float64 = dr # the gap of the particle, set as the particle spacing
    sum_kernel_weight_::Float64 = 0.0 # the sum of kernel weight of the particle, used in density filter
    sum_kernel_weighted_value_::Float64 = 0.0 # the sum of kernel weighted value of the particle, used in density filter
    normal_vec_::RealVector = kVec0 # the normal vector of the particle, used in wall interaction
end

@inline function updateDensityAndPressure!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG # only apply on fluid particles
        p.rho_ += p.drho_ * dt / 2 # half time step, update density
        p.drho_ = 0.0 # reset the density change rate
        p.p_ = c^2 * (p.rho_ - rho_0) # weakly compressible fluid model
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function continuity!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == FLUID_TAG && q.type_ == FLUID_TAG # only apply on fluid and fluid interaction
        kernel_gradient = kernelGradient(r, smooth_kernel) # get the gradient of the kernel function, `Wᵢⱼ`
        # tips: in SPH method, |∇Wᵢⱼ| = ⃗r⋅∇Wᵢⱼ / r, `kernel_gradient` is |∇Wᵢⱼ|, thus, ∇Wᵢⱼ is `kernel_gradient * (p.x_vec_ - q.x_vec_) / r`.
        p.drho_ += q.mass_ * dot(p.v_vec_ - q.v_vec_, p.x_vec_ - q.x_vec_) * kernel_gradient / r # ∑ⱼmⱼ(vᵢ - vⱼ)⋅∇Wᵢⱼ
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function momentum!(p::Particle, q::Particle, r::Float64)::Nothing
    if p.type_ == FLUID_TAG && q.type_ == FLUID_TAG # only apply on fluid and fluid interaction: pressure and viscosity
        # * pressure term
        kernel_value = kernelValue(r, smooth_kernel)
        kernel_gradient = kernelGradient(r, smooth_kernel)
        mean_gap = (p.gap_ + q.gap_) / 2
        p_rho_2 = p.p_ / p.rho_^2 + q.p_ / q.rho_^2 # pᵢ/ρᵢ² + pⱼ/ρⱼ²
        reference_kernel_value = kernelValue(mean_gap, smooth_kernel)
        # in SPH method, to avoid too close distance, a cofficient 0.01 Wᵢⱼ/W(Δp) is added to the pressure term
        p_rho_2 += abs(p_rho_2) * 0.01 * kernel_value / reference_kernel_value
        p.dv_vec_ += -q.mass_ * p_rho_2 * kernel_gradient / r * (p.x_vec_ - q.x_vec_) # -∑ⱼmⱼ(pᵢ/ρᵢ² + pⱼ/ρⱼ² + 0.01 Wᵢⱼ/W(Δp))∇Wᵢⱼ
        # * viscosity term
        mean_mu = 2 * p.mu_ * q.mu_ / (p.mu_ + q.mu_)
        sum_rho = p.rho_ + q.rho_
        viscosity_force = 8 * mean_mu * kernel_gradient * r / sum_rho^2 / (r^2 + 0.01 * h^2) # ∑ⱼ8̄μ/(ρᵢ + ρⱼ)²/(r² + 0.01h²) * r⋅Wᵢⱼ
        p.dv_vec_ += q.mass_ * viscosity_force * (p.v_vec_ - q.v_vec_) # ∑ⱼmⱼ8̄μ/(ρᵢ + ρⱼ)²/(r² + 0.01h²) * r⋅Wᵢⱼ(vᵢ - vⱼ)
        return nothing
    elseif p.type_ == FLUID_TAG && q.type_ == WALL_TAG # only apply on fluid and wall interaction: viscosity, compulsive
        # * viscosity term
        kernel_gradient = kernelGradient(r, smooth_kernel)
        mean_mu = 2 * p.mu_ * q.mu_ / (p.mu_ + q.mu_)
        sum_rho = p.rho_ + q.rho_
        viscosity_force = 8 * mean_mu * kernel_gradient * r / sum_rho^2 / (r^2 + 0.01 * h^2)
        p.dv_vec_ += q.mass_ * viscosity_force * (p.v_vec_ - q.v_vec_) # the same as fluid-fluid interaction, however, wall's mu is much larger
        # * compulsive term
        # * see The paper is from Roger & Dalrmple, 2008, DualSPHysics also adopts this model
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
    if p.type_ == FLUID_TAG # only apply on fluid particles
        p.v_vec_ += (p.dv_vec_ + g) * dt / 2 # half time step, update velocity
        p.dv_vec_ = kVec0 # reset the acceleration
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function updatePosition!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        p.x_vec_ += p.v_vec_ * dt / 2 # half time step, update position
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function densityFilter!(p::Particle, q::Particle, r::Float64)::Nothing # density filter for fluid and fluid interaction
    if p.type_ == FLUID_TAG && q.type_ == FLUID_TAG # only apply on fluid and fluid interaction
        kernel_value = kernelValue(r, smooth_kernel)
        p.sum_kernel_weighted_value_ += q.mass_ * kernel_value
        p.sum_kernel_weight_ += q.mass_ / q.rho_ * kernel_value # ̄ρᵢ = ∑ⱼmⱼWᵢⱼ / ∑ⱼmⱼ/ρⱼWᵢⱼ
        return nothing
    else
        return nothing
    end
    return nothing
end

@inline function densityFilter!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        p.sum_kernel_weighted_value_ += p.mass_ * smooth_kernel.kernel_0_ # don't forget this neighbour includes itself
        p.sum_kernel_weight_ += p.mass_ / p.rho_ * smooth_kernel.kernel_0_
        p.rho_ = p.sum_kernel_weighted_value_ / p.sum_kernel_weight_
        p.p_ = c^2 * (p.rho_ - rho_0) # weakly compressible fluid model, update pressure as long as density is updated
        p.sum_kernel_weight_ = 0.0 # reset the sum of kernel weight
        p.sum_kernel_weighted_value_ = 0.0 # reset the sum of kernel weighted value
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
    particles = Vector{ParticleType}()                 # create particles, first an empty array
    n_along_x = Int64(width / reference_dr |> round)
    n_along_y = Int64(height / reference_dr |> round)
    dx = width / n_along_x
    dy = height / n_along_y
    for i in 1:n_along_x, j in 1:n_along_y
        particle = ParticleType()
        x = x0 + (i - 0.5) * dx # calculate the x position, in the center of each square
        y = y0 + (j - 0.5) * dy # calculate the y position, in the center of each square
        particle.x_vec_ = RealVector(x, y, 0.0)
        modifyOnParticle!(particle) # modify the particle as you wish
        particle.mass_ = particle.rho_ * dx * dy # calculate the mass of the particle
        push!(particles, particle) # add the particle to the array
    end
    return particles
end

const x0 = 0.0
const y0 = 0.0

function initialPressure!(p::Particle)::Nothing
    depth = water_height - p.x_vec_[2] # calculate the depth of the particle
    p.p_ = rho_0 * gravity * depth # calculate the pressure of the particle, p = ρgh
    p.rho_ += p.p_ / p.c_^2 # update the density of the particle
    return nothing
end

particles = Particle[]

fluid_particles =
    createRectangleParticles(Particle, x0, y0, water_width, water_height, dr; modifyOnParticle! = initialPressure!)
append!(particles, fluid_particles)

@inline function bottomParticleModify!(p::Particle)::Nothing
    p.normal_vec_ = kVecY # set the normal vector of the wall to be y-axis
    p.type_ = WALL_TAG # set the type of the wall to be wall
    p.mu_ *= 1000 # set the viscosity coefficient of the wall to be 1000 times of the fluid
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
    p.normal_vec_ = kVecX # set the normal vector of the wall to be x-axis
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
    p.normal_vec_ = -kVecX # set the normal vector of the wall to be -x-axis
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
    p.normal_vec_ = (kVecX + kVecY) / sqrt(2) # set the normal vector of the wall to be (x+y)/√2
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
    p.normal_vec_ = (-kVecX + kVecY) / sqrt(2) # set the normal vector of the wall to be (-x+y)/√2
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
