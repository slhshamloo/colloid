function record!(sim::HPMCSimulation, recorder::TrajectoryRecorder)
    if recorder.cond(sim.timestep)
        if recorder.savetomem
            if isnothing(recorder.trajectory)
                recorder.trajectory = RegularPolygonsTrajectory(
                    sim.particles.sidenum, sim.particles.radius,
                    Vector{Vector{eltype(sim.particles.boxsize)}}(undef, 0),
                    Vector{Matrix{eltype(sim.particles.centers)}}(undef, 0),
                    Vector{Vector{eltype(sim.particles.angles)}}(undef, 0),
                    Vector{typeof(sim.timestep)}(undef, 0))
            end
            push!(recorder.trajectory.boxsizes, Array(sim.particles.boxsize))
            push!(recorder.trajectory.centers, Array(sim.particles.centers))
            push!(recorder.trajectory.angles, Array(sim.particles.angles))
            push!(recorder.trajectory.times, sim.timestep)
        end
        if !isnothing(recorder.filepath)
            Threads.@spawn recordfile!(sim, recorder, Array(sim.particles.centers),
                Array(sim.particles.angles), Array(sim.particles.boxsize))
        end
    end
end

function recordfile!(sim::HPMCSimulation, recorder::TrajectoryRecorder,
        centers::Matrix{<:Real}, angles::Vector{<:Real}, boxsize::Vector{<:Real})
    frame = Threads.@atomic recorder.filecounter += 1
    if frame == 1
        mkdir(recorder.filepath)
        jldopen(recorder.filepath * "/constants.jld2", "a+") do file
            file["seed"] = sim.seed
            file["sidenum"] = sim.particles.sidenum
            file["radius"] = sim.particles.radius
        end
    end
    jldopen(recorder.filepath * "/$frame.jld2", "a+") do file
        file["time"] = sim.timestep
        file["centers"] = Array(sim.particles.centers)
        file["angles"] = Array(sim.particles.angles)
        file["boxsize"] = Array(sim.particles.boxsize)
    end
end

function finalize!(recorder::TrajectoryRecorder)
    jldopen(recorder.filepath * ".jld2", "w+") do masterfile
        jldopen(recorder.filepath * "/constants.jld") do constfile
            masterfile["seed"] = constfile["seed"]
            masterfile["sidenum"] = constfile["seed"]
            masterfile["radius"] = constfile["seed"]
        end
        for frame in 1:recorder.filecounter
            jldopen(recorder.filepath * ".jld2/$frame.jld") do framefile
                masterfile["frame$frame/time"] = framefile["time"]
                masterfile["frame$frame/centers"] = framefile["centers"]
                masterfile["frame$frame/angles"] = framefile["angles"]
                masterfile["frame$frame/boxsize"] = framefile["boxsize"]
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
