function update!(sim::Simulation, tuner::MoveSizeTuner, cell_list::CellList)
    if tuner.cond(sim.timestep)
        translation_acceptance, rotation_acceptance = _get_new_acceptance_rates(sim, tuner)
        _set_tuner_flags!(tuner, translation_acceptance, rotation_acceptance)

        new_move_radius = min(tuner.max_move_radius,
            min(tuner.maxscale, (translation_acceptance + tuner.gamma)
                / (tuner.target_acceptance_rate + tuner.gamma)) * sim.move_radius)
        new_rotation_span = min(tuner.max_rotation_span,
            min(tuner.maxscale, (rotation_acceptance + tuner.gamma)
                / (tuner.target_acceptance_rate + tuner.gamma)) * sim.rotation_span)

        _set_tuner_prev_values!(sim, tuner)
        sim.move_radius, sim.rotation_span = new_move_radius, new_rotation_span
    end
    return cell_list
end

function update!(sim::Simulation, compressor::ForcefulCompressor, cell_list::CellList)
    if _needs_compression(sim, compressor, cell_list)
        if compressor.reached_target
            if sim.gpu
                if iszero(count_overlaps(sim.colloid, cell_list)
                          + count_violations_gpu(sim.colloid, sim.constraints))
                    cell_list = CuCellList(sim.colloid, cell_list.shift)
                    compressor.completed = true
                end
            else
                compressor.completed = true
            end
        else
            pos_scale, lxold, lyold = _apply_compression!(sim, compressor)
            if sim.gpu
                new_cell_list = CuCellList(sim.colloid, cell_list.shift;
                    maxwidth = 0.8 * (sim.colloid.sidenum <= 4 ?
                        2 * sim.colloid.radius : 2 * √2 * sim.colloid.radius))
                violations = count_violations_gpu(sim.colloid, sim.constraints)
            else
                new_cell_list = SeqCellList(sim.colloid)
                violations = count_violations(sim.colloid, sim.constraints)
            end
            if violations + count_overlaps(sim.colloid, new_cell_list) > (
                    compressor.max_overlap_fraction * particle_count(sim.colloid))
                sim.colloid.boxsize[1], sim.colloid.boxsize[2] = lxold, lyold
                sim.colloid.centers ./= pos_scale
            else
                cell_list = new_cell_list
                if (sim.colloid.boxsize[1] == compressor.target_boxsize[1]
                        && sim.colloid.boxsize[2] == compressor.target_boxsize[2])
                    compressor.reached_target = true
                end
            end
        end
    end
    return cell_list
end

function record!(sim::Simulation, recorder::TrajectoryRecorder, cell_list::CellList)
    if recorder.cond(sim.timestep)
        push!(recorder.snapshots, get_snapshot(sim.colloid))
        push!(recorder.times, sim.timestep)
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
            push!(recorder.orders,
                mean(exp.(1im * sim.colloid.sidenum * sim.colloid.angles)))
        end
        push!(recorder.times, sim.timestep)
    end
end

@inline function _get_new_acceptance_rates(sim::Simulation, tuner::MoveSizeTuner)
    acc_trans = sim.accepted_translations - tuner.prev_accepted_translations
    rej_trans = sim.rejected_translations - tuner.prev_rejected_translations
    acc_rot = sim.accepted_rotations - tuner.prev_accepted_rotations
    rej_rot = sim.rejected_rotations - tuner.prev_rejected_rotations
    return acc_trans / (acc_trans + rej_trans), acc_rot / (acc_rot + rej_rot)
end

@inline function _set_tuner_flags!(tuner::MoveSizeTuner,
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

@inline function _set_tuner_prev_values!(sim::Simulation, tuner::MoveSizeTuner)
    tuner.prev_accepted_translations = sim.accepted_translations
    tuner.prev_rejected_translations = sim.rejected_translations
    tuner.prev_accepted_rotations = sim.accepted_rotations
    tuner.prev_rejected_rotations = sim.rejected_rotations
end

@inline function _get_force_compress_dims(sim::Simulation, compressor::ForcefulCompressor)
    scale_factor = max(compressor.minscale,
        1.0 - sim.move_radius / (2 * sim.colloid.radius))

    if sim.colloid.boxsize[1] < compressor.target_boxsize[1]
        lxnew = min(sim.colloid.boxsize[1] / scale_factor, compressor.target_boxsize[1])
    else
        lxnew = max(sim.colloid.boxsize[1] * scale_factor, compressor.target_boxsize[1])
    end

    if sim.colloid.boxsize[2] < compressor.target_boxsize[2]
        lynew = min(sim.colloid.boxsize[2] / scale_factor, compressor.target_boxsize[2])
    else
        lynew = max(sim.colloid.boxsize[2] * scale_factor, compressor.target_boxsize[2])
    end

    return lxnew, lynew
end

@inline function _needs_compression(sim::Simulation, compressor::ForcefulCompressor,
                                   cell_list::CellList)
    return (!compressor.completed && compressor.cond(sim.timestep)
        && !has_overlap(sim.colloid, cell_list)
        && ((!sim.gpu && !has_violation(sim.colloid, sim.constraints))
            || (sim.gpu && count_violations_gpu(sim.colloid, sim.constraints) == 0)))
end

@inline function _apply_compression!(sim::Simulation, compressor::ForcefulCompressor)
    lxnew, lynew = _get_force_compress_dims(sim, compressor)
    lxold, lyold = sim.colloid.boxsize
    
    sim.colloid.boxsize[1], sim.colloid.boxsize[2] = lxnew, lynew
    pos_scale = (lxnew / lxold, lynew / lyold)
    sim.colloid.centers .*= pos_scale
    return pos_scale, lxold, lyold
end
