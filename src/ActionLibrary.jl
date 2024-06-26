#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/23 14:34:01
  @ license: MIT
  @ description: the basic action library, can be used freely in customized SPH solver
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
    libUpdateVelocity!(p::T; dt::Float64 = 0.0, body_force_vec::RealVector = kVec0)

Update the velocity of the particle `p` with the time step `dt` and the body force `body_force_vec`.

# Arguments:

- `p` is a subtype of `AbstractParticle` with properties:
    - `v_vec_::RealVector`: the velocity of the particle.
    - `dv_vec_::RealVector`: the acceleration of the particle.
- `dt` is the time step.
- `body_force_vec` is the body force.

# Returns:
- `Nothing`

# Warning:

This function does not reset `dv_vec_` (acceleration) to zero.
"""
@inline function libUpdateVelocity!(
    p::T;
    dt::Float64 = 0.0,
    body_force_vec::RealVector = kVec0,
)::Nothing where {T <: AbstractParticle}
    p.v_vec_ += (p.dv_vec_ + body_force_vec) * dt
    return nothing
end

"""
    libUpdatePosition!(p::T; dt::Float64 = 0.0)

Update the position of the particle `p` with the time step `dt`.

# Arguments:

- `p` is a subtype of `AbstractParticle` with properties:
    - `x_vec_::RealVector`: the position of the particle.
    - `v_vec_::RealVector`: the velocity of the particle.
    - `dv_vec_::RealVector`: the acceleration of the particle.
- `dt` is the time step.

# Returns:
- `Nothing`

# Warning:

This function resets `dv_vec_` (acceleration) to zero.
"""
@inline function libUpdatePosition!(p::T; dt::Float64 = 0.0)::Nothing where {T <: AbstractParticle}
    p.x_vec_ += p.v_vec_ * dt
    p.dv_vec_ = kVec0
    return nothing
end

"""
    libUpdateDensityAndPressure!(p::T; dt::Float64 = 0.0)

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
@inline function libUpdateVelocityAndPosition!(
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
    libUpdateDensity!(p::T; dt::Float64 = 0.0)

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
@inline function libUpdateDensity!(p::T; dt::Float64 = 0.0)::Nothing where {T <: AbstractParticle}
    p.rho_ += p.drho_ * dt
    p.drho_ = 0.0
    return nothing
end

"""
    libUpdateTemperature!(p::T; dt::Float64 = 0.0)

Update the temperature of the particle `p` with the time step `dt`.

# Arguments:

- `p` is a subtype of `AbstractParticle` with properties:
    - `t_::Float64`: the temperature of the particle.
    - `dt_::Float64`: the change rate of the temperature of the particle.

# Returns:
- `Nothing`
"""
@inline function libUpdateTemperature!(p::T; dt::Float64 = 0.0)::Nothing where {T <: AbstractParticle}
    p.t_ += p.dt_ * dt
    p.dt_ = 0.0
    return nothing
end

"""
    libContinuity!(p::T, q::T, r::Float64; kernel_gradient::Float64 = 0.0)

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
@inline function libContinuity!(
    p::T,
    q::T,
    r::Float64;
    kernel_gradient::Float64 = 0.0,
)::Nothing where {T <: AbstractParticle}
    p.drho_ += q.mass_ * dot(p.v_vec_ - q.v_vec_, p.x_vec_ - q.x_vec_) * kernel_gradient / r
    return nothing
end

"""
    libPressureForce!(p::T, q::T, r::Float64; kernel_value::Float64 = 0.0, kernel_gradient::Float64 = 0.0, reference_kernel_value::Float64 = 1.0)

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
@inline function libPressureForce!(
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
    libMirrorPressureForce!(p::T, q::T, r::Float64; kernel_value::Float64 = 0.0, kernel_gradient::Float64 = 0.0, reference_kernel_value::Float64 = 1.0)

Update the mirror pressure force of the particle `p` with the particle `q` and the distance `r`.

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

Similar to `libPressureForce!`, but the pressure term is given mirrorly by `p` particle, `q` only provides position.
"""
@inline function libMirrorPressureForce!(
    p::T,
    q::T,
    r::Float64;
    kernel_value::Float64 = 0.0,
    kernel_gradient::Float64 = 0.0,
    reference_kernel_value::Float64 = 1.0,
)::Nothing where {T <: AbstractParticle}
    p_rho_2 = p.p_ / p.rho_^2 * 2
    p_rho_2 += abs(p_rho_2) * 0.01 * kernel_value / reference_kernel_value
    p.dv_vec_ += -p.mass_ * p_rho_2 * kernel_gradient / r * (p.x_vec_ - q.x_vec_)
    return nothing
end

"""
    libViscosityForce!(p::T, q::T, r::Float64; kernel_gradient::Float64 = 0.0, h::Float64 = 1.0)

Update the viscosity force of the particle `p` with the particle `q` and the distance `r`.

# Arguments:

- `p` and `q` are subtypes of `AbstractParticle` with properties:
    - `mu_::Float64`: the viscosity of the particle.
    - `rho_::Float64`: the density of the particle.
    - `v_vec_::RealVector`: the velocity of the particle.
    - `dv_vec_::RealVector`: the acceleration of the particle.
    - `mass_::Float64`:
- `r` is the distance between the particles.
- `kernel_gradient` is the gradient of the kernel function.
- `h` is the smoothing length, sometimes is the half of the smoothing length to avoid sigularity.

# Returns:
- `Nothing`

# Tips:

1. in SPH method, harmonic mean of parameters is often used in discontinuous media.
2. formula of viscosity force takes advantage of a trick in Taylor expansion.
"""
@inline function libViscosityForce!(
    p::T,
    q::T,
    r::Float64;
    kernel_gradient::Float64 = 0.0,
    h::Float64 = 0.0,
)::Nothing where {T <: AbstractParticle}
    mean_mu = harmonicMean(p.mu_, q.mu_)
    viscosity_force = 2 * mean_mu * kernel_gradient * r / p.rho_ / q.rho_ / (r^2 + 0.01 * h^2)
    p.dv_vec_ += q.mass_ * viscosity_force * (p.v_vec_ - q.v_vec_)
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
@inline function libCompulsiveForce!(p::T, q::T, r::Float64; h::Float64 = 1.0)::Nothing where {T <: AbstractParticle}
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
@inline function libKernelAverageDensityFilter!(
    p::T,
    q::T,
    r::Float64;
    smooth_kernel::SmoothKernel = kSmoothKernel,
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
@inline function libKernelAverageDensityFilter!(
    p::T;
    smooth_kernel::SmoothKernel = kSmoothKernel,
)::Nothing where {T <: AbstractParticle}
    p.sum_kernel_weighted_value_ += p.mass_ * smooth_kernel.kernel_0_
    p.sum_kernel_weight_ += p.mass_ / p.rho_ * smooth_kernel.kernel_0_
    p.rho_ = p.sum_kernel_weighted_value_ / p.sum_kernel_weight_
    p.sum_kernel_weight_ = 0.0
    p.sum_kernel_weighted_value_ = 0.0
    return nothing
end

"""
    libThermalConduction!(p::T, q::T, r::Float64; kernel_gradient::Float64 = 0.0, h::Float64 = 0.0)

Update the thermal conduction of the particle `p` with the particle `q` and the distance `r`.

# Arguments:

- `p` and `q` are subtypes of `AbstractParticle` with properties:
    - `t_::Float64`: the temperature of the particle.
    - `dt_::Float64`: the change rate of the temperature of the particle.
    - `kappa_::Float64`: the thermal conductivity of the particle.
    - `rho_::Float64`: the density of the particle.
    - `mass_::Float64`:
    - `cp_::Float64`: the specific heat capacity of the particle.
- `r` is the distance between the particles.
- `kernel_gradient` is the gradient of the kernel function.
- `h` is the smoothing length, sometimes is the half of the smoothing length to avoid sigularity.

# Returns:
- `Nothing`

# Tips:

solve such equation: ρcₚ∂T/∂t = ∇·(κ∇T), where κ is hamonic mean of κᵢ and κⱼ to handle discontinuous media.
"""
@inline function libThermalConduction!(
    p::T,
    q::T,
    r::Float64;
    kernel_gradient::Float64 = 0.0,
    h::Float64 = 0.0,
)::Nothing where {T <: AbstractParticle}
    mean_kappa = harmonicMean(p.kappa_, q.kappa_)
    heat = 2 * mean_kappa * r * kernel_gradient / p.rho_ / q.rho_ / (r^2 + 0.01 * h^2) * (p.t_ - q.t_)
    p.dt_ += q.mass_ * heat / p.cp_
    return nothing
end

"""
    libRiemannSolutionZhang(rho_l, rho_r, u_l, u_r, p_l, p_r, c_0, mean_c, mean_v_vec, e_ij; eta=3.0)

Compute the Riemann solution using the Zhang method.

# Arguments
- `rho_l::Float64`: Left density
- `rho_r::Float64`: Right density
- `u_l::Float64`: Left velocity
- `u_r::Float64`: Right velocity
- `p_l::Float64`: Left pressure
- `p_r::Float64`: Right pressure
- `c_0::Float64`: Sound speed
- `mean_c::Float64`: Mean sound speed
- `mean_v_vec::RealVector`: Mean velocity vector
- `e_ij::RealVector`: Unit vector in the direction of the interface

# Optional Arguments
- `eta::Float64 = 3.0`: Dissipation limiter η (default: 3.0)

# Returns
- `v_riemann_vec::RealVector`: Riemann velocity vector
- `p_riemann::Float64`: Riemann pressure

# Tips
- A weakly compressible SPH method based on a low-dissipation Riemann solver, C. Zhang, X.Y. Hu, N.A. Adams, http://dx.doi.org/10.1016/j.jcp.2017.01.027
- Dual-criteria time stepping for weakly compressible smoothed particle hydrodynamics, Chi Zhang, Massoud Rezavand, Xiangyu Hu, https://doi.org/10.1016/j.jcp.2019.109135
- https://www.sphinxsys.org/html/theory.html
"""
@inline function weaklyCompressibleRiemannSolutionZhang(
    rho_l::Float64,
    rho_r::Float64,
    u_l::Float64,
    u_r::Float64,
    p_l::Float64,
    p_r::Float64,
    c_0::Float64, # sound speed
    mean_c::Float64, # mean sound speed
    mean_v_vec::RealVector,
    e_ij::RealVector; # -\vec{r}ᵢⱼ / rᵢⱼ
    eta::Float64 = 3.0, # dissipation limiter η , recommended value is 3.0 here
)::Tuple{RealVector, Float64} # return (v*, p*)
    mean_rho = (rho_l + rho_r) / 2
    mean_u = (u_l + u_r) / 2
    mean_p = (p_l + p_r) / 2
    u_riemann = mean_u + 0.5 * (p_l - p_r) / mean_rho / c_0
    p_riemann = mean_p + 0.5 * (u_l - u_r) * mean_rho * min(eta * max(u_l - u_r, 0.0), mean_c)
    v_riemann_vec = u_riemann * e_ij + (mean_v_vec - mean_u * e_ij)
    return v_riemann_vec, p_riemann
end

@inline function libWeaklyCompressibleRiemannZhang!(
    p::T,
    q::T,
    r::Float64;
    kernel_gradient::Float64 = 0.0,
    c_0::Float64 = 1.0,
    eta::Float64 = 3.0,
)::Nothing where {T <: AbstractParticle}
    e_ij = -(p.x_vec_ - q.x_vec_) / r
    v_riemann_vec, p_riemann = weaklyCompressibleRiemannSolutionZhang(
        p.rho_,
        q.rho_,
        dot(p.v_vec_, e_ij),
        dot(q.v_vec_, e_ij),
        p.p_,
        q.p_,
        c_0,
        (p.c_ + q.c_) / 2,
        (p.v_vec_ + q.v_vec_) / 2,
        e_ij;
        eta = eta,
    )
    p.drho_ += 2 * p.rho_ * q.mass_ / q.rho_ * dot(p.v_vec_ - v_riemann_vec, -e_ij) * kernel_gradient
    p.dv_vec_ += -2 * q.mass_ * p_riemann / p.rho_ / q.rho_ * kernel_gradient * (-e_ij)
    return nothing
end

@inline function libWeaklyCompressibleRiemannWallZhang!(
    p::T,
    q::T,
    r::Float64;
    kernel_gradient::Float64 = 0.0,
    c_0::Float64 = 1.0,
    eta::Float64 = 3.0,
    body_force_vec::RealVector = kVec0,
)::Nothing where {T <: AbstractParticle}
    u_w = -dot(q.normal_vec_, q.v_vec_)
    e_ij = -(p.x_vec_ - q.x_vec_) / r
    u_l = -dot(q.normal_vec_, p.v_vec_)
    u_r = -u_l + 2 * u_w
    p_r = p.p_ + p.rho_ * dot(body_force_vec, q.x_vec_ - p.x_vec_)
    v_riemann_vec, p_riemann = weaklyCompressibleRiemannSolutionZhang(
        p.rho_,
        p.rho_,
        u_l,
        u_r,
        p.p_,
        p_r,
        c_0,
        p.c_,
        (p.v_vec_ + q.v_vec_) / 2,
        # e_ij;
        -q.normal_vec_;
        eta = eta,
    )
    p.drho_ += 2 * p.rho_ * q.mass_ / q.rho_ * dot(p.v_vec_ - v_riemann_vec, -e_ij) * kernel_gradient
    p.dv_vec_ += -2 * q.mass_ * p_riemann / p.rho_ / q.rho_ * kernel_gradient * (-e_ij)
    return nothing
end
