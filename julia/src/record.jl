function record!(sim::Simulation, recorder::TrajectoryRecorder, cell_list::CellList)
    if recorder.cond(sim.timestep)
        if recorder.savetomem
            if isnothing(recorder.trajectory)
                recorder.trajectory = Trajectory(
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
            recordfile!(sim, recorder)
        end
    end
end

function recordfile!(sim, recorder)
    jldopen(recorder.filepath, "a+") do f
        if recorder.filecounter == 0
            f["sidenum"] = sim.colloid.sidenum
            f["radius"] = sim.colloid.radius
        end
        recorder.filecounter += 1
        f["frame$(recorder.filecounter)/time"] = sim.timestep
        f["frame$(recorder.filecounter)/centers"] = Array(sim.colloid.centers)
        f["frame$(recorder.filecounter)/angles"] = Array(sim.colloid.angles)
        f["frame$(recorder.filecounter)/boxsize"] = Array(sim.colloid.boxsize)
    end
    if recorder.safe
        if recorder.filecounter == 1
            mkdir(recorder.filepath[1:end-5] * "/")
            jldopen(recorder.filepath[1:end-5] * "/constants.jld2", "a+") do f
                f["sidenum"] = sim.colloid.sidenum
                f["radius"] = sim.colloid.radius
            end
        end
        jldopen(recorder.filepath[1:end-5]
                * "/$(recorder.filecounter).jld2", "a+") do f
            f["time"] = sim.timestep
            f["centers"] = Array(sim.colloid.centers)
            f["angles"] = Array(sim.colloid.angles)
            f["boxsize"] = Array(sim.colloid.boxsize)
        end
    end
end

function record!(sim::Simulation, recorder::LocalOrderRecorder, cell_list::CellList)
    if recorder.cond(sim.timestep)
        if recorder.type == "katic"
            push!(recorder.orders, katic_order(
                  sim.colloid, cell_list, recorder.typeparams[1]))
        end
        push!(recorder.times, sim.timestep)
    end
end

function record!(sim::Simulation, recorder::GlobalOrderRecorder, cell_list::CellList)
    if recorder.cond(sim.timestep)
        if recorder.type == "orient"
            push!(recorder.orders, mean(
                  exp.(1im * sim.colloid.sidenum * sim.colloid.angles)))
        end
        push!(recorder.times, sim.timestep)
    end
end
