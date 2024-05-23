#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/23 14:34:01
  @ license: MIT
  @ description:
 =#

"""
    harmonicMean(a::Float64, b::Float64)::Float64

Compute the harmonic mean of two numbers.

# Arguments
- `a::Float64`: The first number.
- `b::Float64`: The second number.

# Returns
- `Float64`: The harmonic mean of `a` and `b`.

# Examples
```julia
harmonicMean(1.0, 2.0)
# output: 1.3333333333333333
```
"""
@inline function harmonicMean(a::Float64, b::Float64)::Float64
    return 2 * a * b / (a + b)
end

"""
    updateVelocity!(p::T; dt::Float64 = 0.0, body_force_vec::RealVector = kVec0)

Update the velocity of the particle `p` with the time step `dt` and the body force `body_force_vec`.

# Arguments:

- `p` is a subtype of `AbstractParticle` with properties:
    - `v_vec_::RealVector`: the velocity of the particle.
    - `dv_vec_::RealVector`: the acceleration of the particle.
- `dt` is the time step.
- `body_force_vec` is the body force.

# Returns:
- `Nothing`
"""
@inline function updateVelocity!(
    p::T;
    dt::Float64 = 0.0,
    body_force_vec::RealVector = kVec0,
)::Nothing where {T <: AbstractParticle}
    p.v_vec_ += (p.dv_vec_ + body_force_vec) * dt
    p.dv_vec_ = kVec0
    return nothing
end

"""
    updatePosition!(p::T; dt::Float64 = 0.0)

Update the position of the particle `p` with the time step `dt`.

# Arguments:

- `p` is a subtype of `AbstractParticle` with properties:
    - `x_vec_::RealVector`: the position of the particle.
    - `v_vec_::RealVector`: the velocity of the particle.
- `dt` is the time step.

# Returns:
- `Nothing`
"""
@inline function updatePosition!(p::T; dt::Float64 = 0.0)::Nothing where {T <: AbstractParticle}
    p.x_vec_ += p.v_vec_ * dt
    return nothing
end

"""
    updateDensityAndPressure!(p::T; dt::Float64 = 0.0)

Update the density and pressure of the particle `p` with the time step `dt`.

# Arguments:

- `p` is a subtype of `AbstractParticle` with properties:
    - `rho_::Float64`: the density of the particle.
    - `drho_::Float64`: the change rate of the density of the particle.
    - `p_::Float64`: the pressure of the particle.
- `dt` is the time step.
- `body_force_vec` is the body force.

# Returns:
- `Nothing`
"""
@inline function updateVelocityAndPosition!(
    p::T;
    dt::Float64 = 0.0,
    body_force_vec::RealVector = kVec0,
)::Nothing where {T <: AbstractParticle}
    p.x_vec_ += p.v_vec_ * dt + 0.5 * (p.dv_vec_ + body_force_vec) * dt^2
    p.v_vec_ += (p.dv_vec_ + body_force_vec) * dt
    p.dv_vec_ = kVec0
    return nothing
end

"""
    updateDensity!(p::T; dt::Float64 = 0.0)

Update the density of the particle `p` with the time step `dt`.

# Arguments:

- `p` is a subtype of `AbstractParticle` with properties:
    - `rho_::Float64`: the density of the particle.
    - `drho_::Float64`: the change rate of the density of the particle.
- `dt` is the time step.

# Returns:
- `Nothing`

# Warning:

Usually, pressure should be updated after this function is called, which depends on your thermodynamic model.
"""
@inline function updateDensity!(p::T; dt::Float64 = 0.0)::Nothing where {T <: AbstractParticle}
    p.rho_ += p.drho_ * dt
    p.drho_ = 0.0
    return nothing
end

"""
    continuity!(p::T, q::T, r::Float64; kernel_gradient::Float64 = 0.0)

Update the continuity of the particle `p` and `q` with the distance `r` and the kernel gradient `kernel_gradient`.

# Arguments:

- `p` and `q` are subtypes of `AbstractParticle` with properties:
    - `drho_::Float64`: the change rate of the density of the particle.
    - `v_vec_::RealVector`: the velocity of the particle.
    - `x_vec_::RealVector`: the position of the particle.
    - `mass_::Float64`: the mass of the particle.
- `r` is the distance between the particles.
- `kernel_gradient` is the gradient of the kernel function.

# Returns:
- `Nothing`

# Tips:

In SPH method, |∇Wᵢⱼ| = ⃗r⋅∇Wᵢⱼ / r, `kernel_gradient` is |∇Wᵢⱼ|, thus, ∇Wᵢⱼ is `kernel_gradient * (p.x_vec_ - q.x_vec_) / r`.
"""
@inline function continuity!(
    p::T,
    q::T,
    r::Float64;
    kernel_gradient::Float64 = 0.0,
)::Nothing where {T <: AbstractParticle}
    p.drho_ += q.mass_ * dot(p.v_vec_ - q.v_vec_, p.x_vec_ - q.x_vec_) * kernel_gradient / r
    return nothing
end

"""
    pressureForce!(p::T, q::T, r::Float64; kernel_value::Float64 = 0.0, kernel_gradient::Float64 = 0.0, reference_kernel_value::Float64 = 1.0)

Update the pressure force of the particle `p` with the particle `q` and the distance `r`.

# Arguments:

- `p` and `q` are subtypes of `AbstractParticle` with properties:
    - `p_::Float64`: the pressure of the particle.
    - `rho_::Float64`: the density of the particle.
    - `mass_::Float64`:
    - `x_vec_::RealVector`: the position of the particle.
    - `dv_vec_::RealVector`: the acceleration of the particle.
- `r` is the distance between the particles.
- `kernel_value` is the value of the kernel function.
- `kernel_gradient` is the gradient of the kernel function.
- `reference_kernel_value` is the value of the kernel function at the reference distance.

# Returns:
- `Nothing`

# Tips:

In SPH method, to avoid too close distance, a cofficient 0.01 Wᵢⱼ/W(Δp) is added to the pressure term.
"""
@inline function pressureForce!(
    p::T,
    q::T,
    r::Float64;
    kernel_value::Float64 = 0.0,
    kernel_gradient::Float64 = 0.0,
    reference_kernel_value::Float64 = 1.0,
)::Nothing where {T <: AbstractParticle}
    p_rho_2 = p.p_ / p.rho_^2 + q.p_ / q.rho_^2
    p_rho_2 += abs(p_rho_2) * 0.01 * kernel_value / reference_kernel_value
    p.dv_vec_ += -q.mass_ * p_rho_2 * kernel_gradient / r * (p.x_vec_ - q.x_vec_)
    return nothing
end

"""
    viscosityForce!(p::T, q::T, r::Float64; kernel_gradient::Float64 = 0.0, h::Float64 = 1.0)

Update the viscosity force of the particle `p` with the particle `q` and the distance `r`.

# Arguments:

- `p` and `q` are subtypes of `AbstractParticle` with properties:
    - `mu_::Float64`: the viscosity of the particle.
    - `rho_::Float64`: the density of the particle.
    - `v_vec_::RealVector`: the velocity of the particle.
    - `dv_vec_::RealVector`: the acceleration of the particle.
- `r` is the distance between the particles.
- `kernel_gradient` is the gradient of the kernel function.
- `h` is the smoothing length, sometimes is the half of the smoothing length to avoid sigularity.

# Returns:
- `Nothing`

# Tips:

1. in SPH method, harmonic mean of parameters is often used in discontinuous media.
2. formula of viscosity force takes advantage of a trick in Taylor expansion.
"""
@inline function viscosityForce!(
    p::T,
    q::T,
    r::Float64;
    kernel_gradient::Float64 = 0.0,
    h::Float64 = 1.0,
)::Nothing where {T <: AbstractParticle}
    mean_mu = harmonicMean(p.mu_, q.mu_)
    sum_rho = p.rho_ + q.rho_
    viscosity_force = 8 * mean_mu * kernel_gradient * r / sum_rho^2 / (r^2 + 0.01 * h^2)
    p.dv_vec_ += viscosity_force * (q.v_vec_ - p.v_vec_)
    return nothing
end

"""
    compulsiveForce!(p::T, q::T, r::Float64; h::Float64 = 1.0)

Update the compulsive force of the particle `p` with the particle `q` and the distance `r`.

# Arguments:

- `p` and `q` are subtypes of `AbstractParticle` with properties:
    - `c_::Float64`: the sound speed of the particle.
    - `v_vec_::RealVector`: the velocity of the particle.
    - `dv_vec_::RealVector`: the acceleration of the particle.
    - `gap_::Float64`: the gap between the particles.
    - `normal_vec_::RealVector`: the normal vector of the particle.
- `r` is the distance between the particles.
- `h` is the smoothing length, sometimes is the half of the smoothing length to avoid sigularity.

# Returns:
- `Nothing`

# Tips: 

The paper is from Roger & Dalrmple, 2008, DualSPHysics also adopts this model. Such force blocks the particles from penetrating the wall with given normal vector.
"""
@inline function compulsiveForce!(p::T, q::T, r::Float64; h::Float64 = 1.0)::Nothing where {T <: AbstractParticle}
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
end

"""
    kernelAverageDensityFilter!(p::T, q::T, r::Float64; smooth_kernel::SmoothKernel)

Update the kernel average density filter of the particle `p` with the particle `q` and the distance `r`.

# Arguments:

- `p` and `q` are subtypes of `AbstractParticle` with properties:
    - `mass_::Float64`: the mass of the particle.
    - `rho_::Float64`: the density of the particle.
    - `sum_kernel_weighted_value_::Float64`: the sum of the kernel weighted value of the particle.
    - `sum_kernel_weight_::Float64`: the sum of the kernel weight of the particle.
- `r` is the distance between the particles.
- `smooth_kernel` is the smoothing kernel.

# Returns:
- `Nothing`

# Tips:

ρᵢ = ∑ⱼmⱼWᵢⱼ / ∑ⱼmⱼ/ρⱼWᵢⱼ, see also `kernelAverageDensityFilter!(p::T, smooth_kernel::SmoothKernel)`.
"""
@inline function kernelAverageDensityFilter!(
    p::T,
    q::T,
    r::Float64,
    smooth_kernel::SmoothKernel,
)::Nothing where {T <: AbstractParticle}
    kernel_value = kernelValue(r, smooth_kernel)
    p.sum_kernel_weighted_value_ += q.mass_ * kernel_value
    p.sum_kernel_weight_ += q.mass_ / q.rho_ * kernel_value
    return nothing
end

"""
    kernelAverageDensityFilter!(p::T; smooth_kernel::SmoothKernel)

Update the kernel average density filter of the particle `p` with the smoothing kernel.

# Arguments:

- `p` is a subtype of `AbstractParticle` with properties:
    - `mass_::Float64`:
    - `rho_::Float64`:
    - `sum_kernel_weighted_value_::Float64`:
    - `sum_kernel_weight_::Float64`:
- `smooth_kernel` is the smoothing kernel.

# Returns:
- `Nothing`

# Tips:

ρᵢ = ∑ⱼmⱼWᵢⱼ / ∑ⱼmⱼ/ρⱼWᵢⱼ, see also `kernelAverageDensityFilter!(p::T, q::T, r::Float64; smooth_kernel::SmoothKernel)`.
"""
@inline function kernelAverageDensityFilter!(p::T, smooth_kernel::SmoothKernel)::Nothing where {T <: AbstractParticle}
    p.sum_kernel_weighted_value_ += p.mass_ * smooth_kernel.kernel_0_
    p.sum_kernel_weight_ += p.mass_ / p.rho_ * smooth_kernel.kernel_0_
    p.rho_ = p.sum_kernel_weighted_value_ / p.sum_kernel_weight_
    p.sum_kernel_weight_ = 0.0
    p.sum_kernel_weighted_value_ = 0.0
    return nothing
end