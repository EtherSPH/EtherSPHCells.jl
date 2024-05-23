#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/22 21:38:44
  @ license: MIT
  @ description:
 =#

using ProgressBars
include("collapse_dry_base.jl")

start_point = RealVector(-h, -h, 0.0) # calculation domain start point, to set the background cell list
end_point = RealVector(box_width + h, box_height + h, 0.0) # calculation domain end point, to set the background cell list
system = ParticleSystem(Particle, h, start_point, end_point) # create a particle system
append!(system.particles_, particles) # add particles to the system

vtp_io = VTPIO() # create a VTPIO object, to write the results to `.vtp``
@inline getPressure(p::Particle)::Float64 = p.p_ # get the pressure of the particle
@inline getVelocity(p::Particle)::RealVector = p.v_vec_ # get the velocity of the particle
@inline getNormalVector(p::Particle)::RealVector = p.normal_vec_ # get the normal vector of the particle
addScalar!(vtp_io, "Pressure", getPressure) # add the pressure to the VTPIO object, make it know to write the pressure to `.vtp`
addVector!(vtp_io, "Velocity", getVelocity) # add the velocity to the VTPIO object, make it know to write the velocity to `.vtp`
addVector!(vtp_io, "Normal", getNormalVector) # add the normal vector to the VTPIO object, make it know to write the normal vector to `.vtp`
vtp_io.step_digit_ = 4 # set the step digit to 4, 0001.vtp, 0002.vtp, ...0010.vtp, ... 0100.vtp, ...
vtp_io.file_name_ = "collapse_dry" # set the file name to "collapse_dry"
vtp_io.output_path_ = "demo/results/collapse_dry_results" # set the output path to "demo/results/collapse_dry_results"

function main()::Nothing
    assurePathExist(vtp_io) # assure the path exist, if not, create it
    t = 0.0 # set the initial time to 0.0
    saveVTP(vtp_io, system, 0, t) # save the initial state to `.vtp`
    updateBackgroundCellList!(system) # update the background cell list
    applyInteraction!(system, momentum!) # apply the momentum interaction
    for step in ProgressBar(1:round(Int, t_end / dt)) # loop over the time steps
        applySelfaction!(system, updateVelocity!) # accelerate the particles
        applySelfaction!(system, updatePosition!) # move the particles
        updateBackgroundCellList!(system) # update the background cell list
        applyInteraction!(system, continuity!) # continuity interaction
        applySelfaction!(system, updateDensityAndPressure!) # update the density and pressure
        applySelfaction!(system, updatePosition!) # move the particles
        updateBackgroundCellList!(system) # update the background cell list
        applyInteraction!(system, momentum!) # momentum interaction
        applySelfaction!(system, updateVelocity!) # accelerate the particles
        if step % round(Int, output_dt / dt) == 0 # output the results
            saveVTP(vtp_io, system, step, t) # save the results to `.vtp`
        end
        if step % round(Int, density_filter_dt / dt) == 0
            applyInteraction!(system, densityFilter!) # need to implement the density filter
            applySelfaction!(system, densityFilter!) # don't forget itself
        end
        t += dt
    end
    return nothing
end
