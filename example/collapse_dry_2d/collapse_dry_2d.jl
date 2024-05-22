#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/22 21:38:44
  @ license: MIT
  @ description:
 =#

using ProgressBars
include("collapse_dry_2d_base.jl")

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
vtp_io.file_name_ = "collapse_dry"
vtp_io.output_path_ = "example/collapse_dry_2d/collapse_dry_2d_results"
vtp_io

function main()::Nothing
    assurePathExist(vtp_io)
    t = 0.0
    saveVTP(vtp_io, system, 0, t)
    updateBackgroundCellList!(system)
    applyInteraction!(system, momentum!)
    for step in ProgressBar(1:round(Int, t_end / dt))
        applySelfaction!(system, updateVelocity!)
        applySelfaction!(system, updatePosition!)
        updateBackgroundCellList!(system)
        applyInteraction!(system, continuity!)
        applySelfaction!(system, updateDensityAndPressure!)
        applySelfaction!(system, updatePosition!)
        updateBackgroundCellList!(system)
        applyInteraction!(system, momentum!)
        applySelfaction!(system, updateVelocity!)
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
