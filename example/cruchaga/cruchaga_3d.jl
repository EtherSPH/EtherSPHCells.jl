#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/23 23:58:04
  @ license: MIT
  @ description:
 =#

# M.A. Cruchaga

using EtherSPHCells
using Parameters
using ProgressBars

const dim = 3
const dr = 0.114 / 40
const gap = dr
const h = 3 * dr

const smooth_kernel = SmoothKernel(h, dim, WendlandC2)

const water_x_len = 0.114
const water_y_len = 0.114
const water_z_len = 0.228
const box_x_len = 0.42
const box_y_len = 0.44
const box_z_len = 0.228
const wall_width = h

const rho_0 = 1000.0
const mass = rho_0 * dr^dim
const gravity = 9.8
const c = 10 * sqrt(2 * gravity * water_y_len)
const g = RealVector(0.0, -gravity, 0.0)
const mu = 1e-3
const mu_wall = mu * 1000
const nu = mu / rho_0
const FLUID_TAG = 1
const WALL_TAG = 2

@inline function getPressureFromDensity(rho::Float64)::Float64
    return c^2 * (rho - rho_0)
end

@inline function getDensityFromPressure(p::Float64)::Float64
    return p / c^2 + rho_0
end

const dt = 0.1 * h / c
const t_end = 2.0
const output_dt = 100 * dt
const density_filter_dt = 20 * dt

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

@inline function updateVelocityAndPosition!(p::Particle)::Nothing
    if p.type_ == FLUID_TAG
        libUpdateVelocityAndPosition!(p; dt = dt, body_force_vec = g)
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

function createCubeParticles(
    ParticleType::DataType,
    x0::Float64,
    y0::Float64,
    z0::Float64,
    x_len::Float64,
    y_len::Float64,
    z_len::Float64,
    reference_dr::Float64;
    modifyOnParticle!::Function = EtherSPHCells.noneFunction!,
)::Vector{ParticleType}
    particles = Vector{ParticleType}()
    n_along_x = Int64(round(x_len / reference_dr))
    n_along_y = Int64(round(y_len / reference_dr))
    n_along_z = Int64(round(z_len / reference_dr))
    dx = x_len / n_along_x
    dy = y_len / n_along_y
    dz = z_len / n_along_z
    for i in 1:n_along_x, j in 1:n_along_y, k in 1:n_along_z
        particle = ParticleType()
        x = x0 + (i - 0.5) * dx
        y = y0 + (j - 0.5) * dy
        z = z0 + (k - 0.5) * dz
        particle.x_vec_ = RealVector(x, y, z)
        modifyOnParticle!(particle)
        particle.mass_ = particle.rho_ * dx * dy * dz
        push!(particles, particle)
    end
    return particles
end

const x0 = 0.0
const y0 = 0.0
const z0 = 0.0

function initialPressure!(p::Particle)::Nothing
    depth = water_y_len - p.x_vec_[2]
    p.p_ = rho_0 * gravity * depth
    p.rho_ = getDensityFromPressure(p.p_)
    return nothing
end

particles = Particle[]

fluid_particles = createCubeParticles(
    Particle,
    x0,
    y0,
    z0,
    water_x_len,
    water_y_len,
    water_z_len,
    dr;
    modifyOnParticle! = initialPressure!,
)
append!(particles, fluid_particles)

@inline function xyBottomParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = kVecZ
    p.mu_ = mu_wall
    return nothing
end
xy_bottom_particles = createCubeParticles(
    Particle,
    x0,
    y0,
    z0 - wall_width,
    box_x_len,
    box_y_len,
    wall_width,
    dr;
    modifyOnParticle! = xyBottomParticleModify!,
)
append!(particles, xy_bottom_particles)

@inline function xyTopParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = -kVecZ
    p.mu_ = mu_wall
    return nothing
end
xy_top_particles = createCubeParticles(
    Particle,
    x0,
    y0,
    z0 + box_z_len,
    box_x_len,
    box_y_len,
    wall_width,
    dr;
    modifyOnParticle! = xyTopParticleModify!,
)
append!(particles, xy_top_particles)

@inline function xzFrontParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = kVecY
    p.mu_ = mu_wall
    return nothing
end
xz_front_particles = createCubeParticles(
    Particle,
    x0,
    y0 - wall_width,
    z0,
    box_x_len,
    wall_width,
    box_z_len,
    dr;
    modifyOnParticle! = xzFrontParticleModify!,
)
append!(particles, xz_front_particles)

@inline function xzBackParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = -kVecY
    p.mu_ = mu_wall
    return nothing
end
xz_back_particles = createCubeParticles(
    Particle,
    x0,
    y0 + box_y_len,
    z0,
    box_x_len,
    wall_width,
    box_z_len,
    dr;
    modifyOnParticle! = xzBackParticleModify!,
)
append!(particles, xz_back_particles)

@inline function yzLeftParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = kVecX
    p.mu_ = mu_wall
    return nothing
end
yz_left_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0,
    z0,
    wall_width,
    box_y_len,
    box_z_len,
    dr;
    modifyOnParticle! = yzLeftParticleModify!,
)
append!(particles, yz_left_particles)

@inline function yzRightParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = -kVecX
    p.mu_ = mu_wall
    return nothing
end
yz_right_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0,
    z0,
    wall_width,
    box_y_len,
    box_z_len,
    dr;
    modifyOnParticle! = yzRightParticleModify!,
)
append!(particles, yz_right_particles)

@inline function XminYminZminCornerParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecX + kVecY + kVecZ) / sqrt(3)
    p.mu_ = mu_wall
    return nothing
end
xmin_ymin_zmin_corner_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0 - wall_width,
    z0 - wall_width,
    wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XminYminZminCornerParticleModify!,
)
append!(particles, xmin_ymin_zmin_corner_particles)

@inline function XminYminZmaxCornerParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecX + kVecY - kVecZ) / sqrt(3)
    p.mu_ = mu_wall
    return nothing
end
xmin_ymin_zmax_corner_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0 - wall_width,
    z0 + box_z_len,
    wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XminYminZmaxCornerParticleModify!,
)
append!(particles, xmin_ymin_zmax_corner_particles)

@inline function XminYmaxZminCornerParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecX - kVecY + kVecZ) / sqrt(3)
    p.mu_ = mu_wall
    return nothing
end
xmin_ymax_zmin_corner_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0 + box_y_len,
    z0 - wall_width,
    wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XminYmaxZminCornerParticleModify!,
)
append!(particles, xmin_ymax_zmin_corner_particles)

@inline function XminYmaxZmaxCornerParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecX - kVecY - kVecZ) / sqrt(3)
    p.mu_ = mu_wall
    return nothing
end
xmin_ymax_zmax_corner_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0 + box_y_len,
    z0 + box_z_len,
    wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XminYmaxZmaxCornerParticleModify!,
)
append!(particles, xmin_ymax_zmax_corner_particles)

@inline function XmaxYminZminCornerParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecX + kVecY + kVecZ) / sqrt(3)
    p.mu_ = mu_wall
    return nothing
end
xmax_ymin_zmin_corner_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0 - wall_width,
    z0 - wall_width,
    wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XmaxYminZminCornerParticleModify!,
)
append!(particles, xmax_ymin_zmin_corner_particles)

@inline function XmaxYminZmaxCornerParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecX + kVecY - kVecZ) / sqrt(3)
    p.mu_ = mu_wall
    return nothing
end
xmax_ymin_zmax_corner_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0 - wall_width,
    z0 + box_z_len,
    wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XmaxYminZmaxCornerParticleModify!,
)
append!(particles, xmax_ymin_zmax_corner_particles)

@inline function XmaxYmaxZminCornerParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecX - kVecY + kVecZ) / sqrt(3)
    p.mu_ = mu_wall
    return nothing
end
xmax_ymax_zmin_corner_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0 + box_y_len,
    z0 - wall_width,
    wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XmaxYmaxZminCornerParticleModify!,
)
append!(particles, xmax_ymax_zmin_corner_particles)

@inline function XmaxYmaxZmaxCornerParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecX - kVecY - kVecZ) / sqrt(3)
    p.mu_ = mu_wall
    return nothing
end
xmax_ymax_zmax_corner_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0 + box_y_len,
    z0 + box_z_len,
    wall_width,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XmaxYmaxZmaxCornerParticleModify!,
)
append!(particles, xmax_ymax_zmax_corner_particles)

@inline function XaxisYminZminEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecY + kVecZ) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
xaxis_ymin_zmin_edge_particles = createCubeParticles(
    Particle,
    x0,
    y0 - wall_width,
    z0 - wall_width,
    box_x_len,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XaxisYminZminEdgeParticleModify!,
)
append!(particles, xaxis_ymin_zmin_edge_particles)

@inline function XaxisYminZmaxEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecY - kVecZ) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
xaxis_ymin_zmax_edge_particles = createCubeParticles(
    Particle,
    x0,
    y0 - wall_width,
    z0 + box_z_len,
    box_x_len,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XaxisYminZmaxEdgeParticleModify!,
)
append!(particles, xaxis_ymin_zmax_edge_particles)

@inline function XaxisYmaxZminEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecY + kVecZ) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
xaxis_ymax_zmin_edge_particles = createCubeParticles(
    Particle,
    x0,
    y0 + box_y_len,
    z0 - wall_width,
    box_x_len,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XaxisYmaxZminEdgeParticleModify!,
)
append!(particles, xaxis_ymax_zmin_edge_particles)

@inline function XaxisYmaxZmaxEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecY - kVecZ) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
xaxis_ymax_zmax_edge_particles = createCubeParticles(
    Particle,
    x0,
    y0 + box_y_len,
    z0 + box_z_len,
    box_x_len,
    wall_width,
    wall_width,
    dr;
    modifyOnParticle! = XaxisYmaxZmaxEdgeParticleModify!,
)
append!(particles, xaxis_ymax_zmax_edge_particles)

@inline function YaxisXminZminEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecX + kVecZ) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
yaxis_xmin_zmin_edge_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0,
    z0 - wall_width,
    wall_width,
    box_y_len,
    wall_width,
    dr;
    modifyOnParticle! = YaxisXminZminEdgeParticleModify!,
)
append!(particles, yaxis_xmin_zmin_edge_particles)

@inline function YaxisXminZmaxEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecX - kVecZ) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
yaxis_xmin_zmax_edge_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0,
    z0 + box_z_len,
    wall_width,
    box_y_len,
    wall_width,
    dr;
    modifyOnParticle! = YaxisXminZmaxEdgeParticleModify!,
)
append!(particles, yaxis_xmin_zmax_edge_particles)

@inline function YaxisXmaxZminEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecX + kVecZ) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
yaxis_xmax_zmin_edge_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0,
    z0 - wall_width,
    wall_width,
    box_y_len,
    wall_width,
    dr;
    modifyOnParticle! = YaxisXmaxZminEdgeParticleModify!,
)
append!(particles, yaxis_xmax_zmin_edge_particles)

@inline function YaxisXmaxZmaxEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecX - kVecZ) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
yaxis_xmax_zmax_edge_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0,
    z0 + box_z_len,
    wall_width,
    box_y_len,
    wall_width,
    dr;
    modifyOnParticle! = YaxisXmaxZmaxEdgeParticleModify!,
)
append!(particles, yaxis_xmax_zmax_edge_particles)

@inline function ZaxisXminYminEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecX + kVecY) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
zaxis_xmin_ymin_edge_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0 - wall_width,
    z0,
    wall_width,
    wall_width,
    box_z_len,
    dr;
    modifyOnParticle! = ZaxisXminYminEdgeParticleModify!,
)
append!(particles, zaxis_xmin_ymin_edge_particles)

@inline function ZaxisXminYmaxEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (kVecX - kVecY) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
zaxis_xmin_ymax_edge_particles = createCubeParticles(
    Particle,
    x0 - wall_width,
    y0 + box_y_len,
    z0,
    wall_width,
    wall_width,
    box_z_len,
    dr;
    modifyOnParticle! = ZaxisXminYmaxEdgeParticleModify!,
)
append!(particles, zaxis_xmin_ymax_edge_particles)

@inline function ZaxisXmaxYminEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecX + kVecY) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
zaxis_xmax_ymin_edge_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0 - wall_width,
    z0,
    wall_width,
    wall_width,
    box_z_len,
    dr;
    modifyOnParticle! = ZaxisXmaxYminEdgeParticleModify!,
)
append!(particles, zaxis_xmax_ymin_edge_particles)

@inline function ZaxisXmaxYmaxEdgeParticleModify!(p::Particle)::Nothing
    p.type_ = WALL_TAG
    p.normal_vec_ = (-kVecX - kVecY) / sqrt(2)
    p.mu_ = mu_wall
    return nothing
end
zaxis_xmax_ymax_edge_particles = createCubeParticles(
    Particle,
    x0 + box_x_len,
    y0 + box_y_len,
    z0,
    wall_width,
    wall_width,
    box_z_len,
    dr;
    modifyOnParticle! = ZaxisXmaxYmaxEdgeParticleModify!,
)
append!(particles, zaxis_xmax_ymax_edge_particles)

start_point = RealVector(-h, -h, -h)
end_point = RealVector(box_x_len + h, box_y_len + h, box_z_len + h)
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
vtp_io.file_name_ = "cruchaga_3d"
vtp_io.output_path_ = "example/results/cruchaga/cruchaga_3d_results"
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
