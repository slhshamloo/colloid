function update!(sim::HPMCSimulation, tuner::MoveSizeTuner)
    if tuner.cond(sim.timestep)
        translation_acceptance, rotation_acceptance = get_new_acceptance_rates(sim, tuner)
        set_tuner_flags!(tuner, translation_acceptance, rotation_acceptance)

        new_move_radius = min(tuner.max_move_radius,
            min(tuner.maxscale, (translation_acceptance + tuner.gamma)
                / (tuner.target_acceptance_rate + tuner.gamma)) * sim.move_radius)
        new_rotation_span = min(tuner.max_rotation_span,
            min(tuner.maxscale, (rotation_acceptance + tuner.gamma)
                / (tuner.target_acceptance_rate + tuner.gamma)) * sim.rotation_span)

        set_tuner_prev_values!(sim, tuner)
        sim.move_radius, sim.rotation_span = new_move_radius, new_rotation_span
    end
end

function update!(sim::HPMCSimulation, tuner::BoxMoveTuner)
    if tuner.cond(sim.timestep)
        acceptance = get_new_acceptance_rates(tuner)
        set_tuner_flags!(tuner, acceptance)
        @. tuner.boxmover.change = min(tuner.max_change,
            min((acceptance + tuner.gamma) / (tuner.target_acceptance_rate + tuner.gamma),
                tuner.maxscale,) * tuner.boxmover.change)
        tuner.prev_accepted_moves .= tuner.boxmover.accepted_moves
    end
end

function update!(sim::HPMCSimulation, tuner::AreaUpdateTuner)
    if tuner.cond(sim.timestep)
        acc_moves = tuner.areaupdater.accepted_moves - tuner.prev_accepted_moves
        rej_moves = tuner.areaupdater.rejected_moves - tuner.prev_rejected_moves
        acceptance_rate = acc_moves / (acc_moves + rej_moves)
        if abs(acceptance_rate - tuner.target_acceptance_rate) <= tuner.tollerance
            if tuner.prev_tuned
                tuner.tuned = true
            else
                tuner.prev_tuned = true
            end
        end

        tuner.areaupdater.areachange = min(tuner.max_move_size,
            min(tuner.maxscale, (acceptance_rate + tuner.gamma)
                / (tuner.target_acceptance_rate + tuner.gamma))
                * tuner.areaupdater.areachange)

        tuner.prev_accepted_moves = tuner.areaupdater.accepted_moves
        tuner.prev_rejected_moves = tuner.areaupdater.rejected_moves
    end
end

@inline function get_new_acceptance_rates(sim::HPMCSimulation, tuner::MoveSizeTuner)
    acc_trans = sim.accepted_translations - tuner.prev_accepted_translations
    rej_trans = sim.rejected_translations - tuner.prev_rejected_translations
    acc_rot = sim.accepted_rotations - tuner.prev_accepted_rotations
    rej_rot = sim.rejected_rotations - tuner.prev_rejected_rotations
    return acc_trans / (acc_trans + rej_trans), acc_rot / (acc_rot + rej_rot)
end

@inline function get_new_acceptance_rates(tuner::BoxMoveTuner)
    accs = tuner.boxmover.accepted_moves - tuner.prev_accepted_moves
    rejs = tuner.boxmover.rejected_moves - tuner.prev_rejected_moves
    return [acc + rej == 0 ? 1.0 : acc / (acc + rej) for (acc, rej) in zip(accs, rejs)]
end

@inline function set_tuner_flags!(tuner::MoveSizeTuner,
        translation_acceptance::Real, rotation_acceptance::Real)
    if abs(translation_acceptance - tuner.target_acceptance_rate) <= tuner.tollerance
        if tuner.prev_translation_tuned
            tuner.translation_tuned = true
        else
            tuner.prev_translation_tuned = true
        end
    end
    if abs(rotation_acceptance - tuner.target_acceptance_rate) <= tuner.tollerance
        if tuner.prev_rotation_tuned
            tuner.rotation_tuned = true
        else
            tuner.prev_rotation_tuned = true
        end
    end
end

@inline function set_tuner_flags!(tuner::BoxMoveTuner, acceptance::AbstractVector)
    tuned_now = abs.(acceptance .- tuner.target_acceptance_rate) .<= tuner.tollerance
    tuner.tuned .= tuned_now .& tuner.prev_tuned
    tuner.prev_tuned .= tuned_now
end

@inline function set_tuner_prev_values!(sim::HPMCSimulation, tuner::MoveSizeTuner)
    tuner.prev_accepted_translations = sim.accepted_translations
    tuner.prev_rejected_translations = sim.rejected_translations
    tuner.prev_accepted_rotations = sim.accepted_rotations
    tuner.prev_rejected_rotations = sim.rejected_rotations
end
