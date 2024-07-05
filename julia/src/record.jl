function record!(sim::HPMCSimulation, recorder::TrajectoryRecorder)
    if recorder.cond(sim.timestep)
        if recorder.savetomem
            if isnothing(recorder.trajectory)
                recorder.trajectory = RegularPolygonsTrajectory(
                    sim.particles.sidenum, sim.particles.radius,
                    Vector{eltype(sim.particles.boxshear)}(undef, 0),
                    Vector{Vector{eltype(sim.particles.boxsize)}}(undef, 0),
                    Vector{Matrix{eltype(sim.particles.centers)}}(undef, 0),
                    Vector{Vector{eltype(sim.particles.angles)}}(undef, 0),
                    Vector{typeof(sim.timestep)}(undef, 0))
            end
            CUDA.@allowscalar push!(recorder.trajectory.boxshears, sim.particles.boxshear[])
            push!(recorder.trajectory.boxsizes, Array(sim.particles.boxsize))
            push!(recorder.trajectory.centers, Array(sim.particles.centers))
            push!(recorder.trajectory.angles, Array(sim.particles.angles))
            push!(recorder.trajectory.times, sim.timestep)
        end
        if !isnothing(recorder.filepath)
            CUDA.@allowscalar recordfile!(sim, recorder, sim.timestep,
                Array(sim.particles.centers), Array(sim.particles.angles),
                Array(sim.particles.boxsize), sim.particles.boxshear[])
        end
    end
end

function recordfile!(sim::HPMCSimulation, recorder::TrajectoryRecorder, timestep::Integer,
        centers::Matrix{<:Real}, angles::Vector{<:Real},
        boxsize::Vector{<:Real}, boxshear::Real)
    recorder.filecounter += 1
    if recorder.filecounter == 1
        rm(recorder.filepath, recursive=true, force=true)
        mkdir(recorder.filepath)
        jldopen(recorder.filepath * "/constants.jld2", "a+") do file
            file["seed"] = sim.seed
            file["sidenum"] = sim.particles.sidenum
            file["radius"] = sim.particles.radius
        end
    end
    jldopen(recorder.filepath * "/$(recorder.filecounter).jld2", "a+") do file
        file["time"] = timestep
        file["centers"] = centers
        file["angles"] = angles
        file["boxsize"] = boxsize
        file["boxshear"] = boxshear
    end
end

"""
    finalize!(recorder)

Save every frame of the recorder trajectory in on file and remove temporary files.
"""
function finalize!(recorder::TrajectoryRecorder)
    jldopen(recorder.filepath * ".jld2", "w+") do masterfile
        jldopen(recorder.filepath * "/constants.jld2") do constfile
            masterfile["seed"] = constfile["seed"]
            masterfile["sidenum"] = constfile["sidenum"]
            masterfile["radius"] = constfile["radius"]
        end
        for frame in 1:recorder.filecounter
            jldopen(recorder.filepath * "/$frame.jld2") do framefile
                masterfile["frame$frame/time"] = framefile["time"]
                masterfile["frame$frame/centers"] = framefile["centers"]
                masterfile["frame$frame/angles"] = framefile["angles"]
                masterfile["frame$frame/boxsize"] = framefile["boxsize"]
                masterfile["frame$frame/boxshear"] = framefile["boxshear"]
            end
        end
    end
    rm(recorder.filepath, recursive=true)
end

function record!(sim::HPMCSimulation, recorder::LocalParamRecorder)
    if recorder.cond(sim.timestep)
        if recorder.type == "katic"
            if eltype(eltype(recorder.values)) <: Complex
                numtype = eltype(eltype(recorder.values)) == ComplexF32 ? Float32 : Float64
            end
            push!(recorder.values, katic_order(
                  sim.particles, sim.cell_list, recorder.typeparams[1]; numtype=numtype))
        end
        push!(recorder.times, sim.timestep)
    end
end

function record!(sim::HPMCSimulation, recorder::GlobalParamRecorder)
    if recorder.cond(sim.timestep)
        if recorder.type == "orient"
            push!(recorder.values, mean(
                  exp.(1im * sim.particles.sidenum * sim.particles.angles)))
        end
        push!(recorder.times, sim.timestep)
    end
end
