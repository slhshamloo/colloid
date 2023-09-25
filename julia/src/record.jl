function record!(sim::ColloidSim, recorder::TrajectoryRecorder,
                 cell_list::Union{Nothing, CellList} = nothing)
    if recorder.cond(sim.timestep)
        if recorder.savetomem
            if isnothing(recorder.trajectory)
                recorder.trajectory = ColloidTrajectory(
                    sim.colloid.sidenum, sim.colloid.radius,
                    Vector{Vector{eltype(sim.colloid.boxsize)}}(undef, 0),
                    Vector{Matrix{eltype(sim.colloid.centers)}}(undef, 0),
                    Vector{Vector{eltype(sim.colloid.angles)}}(undef, 0),
                    Vector{typeof(sim.timestep)}(undef, 0))
            end
            push!(recorder.trajectory.boxsizes, Array(sim.colloid.boxsize))
            push!(recorder.trajectory.centers, Array(sim.colloid.centers))
            push!(recorder.trajectory.angles, Array(sim.colloid.angles))
            push!(recorder.trajectory.times, sim.timestep)
        end
        if !isnothing(recorder.filepath)
            Threads.@spawn recordfile!(sim, recorder, Array(sim.colloid.centers),
                Array(sim.colloid.angles), Array(sim.colloid.boxsize))
        end
    end
end

function recordfile!(sim::ColloidSim, recorder::TrajectoryRecorder,
        centers::Matrix{<:Real}, angles::Vector{<:Real}, boxsize::Vector{<:Real})
    frame = Threads.@atomic recorder.filecounter += 1
    if frame == 1
        mkdir(recorder.filepath)
        jldopen(recorder.filepath * "/constants.jld2", "a+") do file
            file["seed"] = sim.seed
            file["sidenum"] = sim.colloid.sidenum
            file["radius"] = sim.colloid.radius
        end
    end
    jldopen(recorder.filepath * "/$frame.jld2", "a+") do file
        file["time"] = sim.timestep
        file["centers"] = Array(sim.colloid.centers)
        file["angles"] = Array(sim.colloid.angles)
        file["boxsize"] = Array(sim.colloid.boxsize)
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

function record!(sim::ColloidSim, recorder::LocalParamRecorder, cell_list::CellList)
    if recorder.cond(sim.timestep)
        if recorder.type == "katic"
            if eltype(eltype(recorder.values)) <: Complex
                numtype = eltype(eltype(recorder.values)) == ComplexF32 ? Float32 : Float64 
            end
            push!(recorder.values, katic_order(
                  sim.colloid, cell_list, recorder.typeparams[1]; numtype=numtype))
        end
        push!(recorder.times, sim.timestep)
    end
end

function record!(sim::ColloidSim, recorder::GlobalParamRecorder,
                 cell_list::Union{Nothing, CellList} = nothing)
    if recorder.cond(sim.timestep)
        if recorder.type == "orient"
            push!(recorder.values, mean(
                  exp.(1im * sim.colloid.sidenum * sim.colloid.angles)))
        end
        push!(recorder.times, sim.timestep)
    end
end
