function record!(sim::Simulation, recorder::TrajectoryRecorder)
    if recorder.cond(sim.timestep)
        push!(recorder.snapshots, get_snapshot(sim.colloid))
    end
end
