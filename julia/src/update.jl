function update!(sim::ColloidSim, tuner::MoveSizeTuner)
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

function update!(sim::ColloidSim, compressor::ForcefulCompressor)
    if needs_compression(sim, compressor)
        if compressor.reached_target
            set_complete_flag!(sim, compressor)
        else
            pos_scale, lxold, lyold = apply_compression!(sim, compressor)
            if sim.gpu
                new_cell_list = CuCellList(sim.colloid, sim.cell_list.shift,
                    maxwidth=minimum(pos_scale)*get_maxwidth(sim.colloid))
                violations = count_violations_gpu(sim.colloid, sim.constraints)
            else
                new_cell_list = SeqCellList(sim.colloid)
                violations = count_violations(sim.colloid, sim.constraints)
            end
            if violations + count_overlaps(sim.colloid, new_cell_list) > (
                    compressor.max_overlap_fraction * pcount(sim.colloid))
                CUDA.@allowscalar sim.colloid.boxsize[1], sim.colloid.boxsize[2] = (
                    lxold, lyold)
                sim.colloid.centers ./= pos_scale
            else
                sim.cell_list = new_cell_list
                CUDA.allowscalar() do
                    if (sim.colloid.boxsize[1] == compressor.target_boxsize[1]
                            && sim.colloid.boxsize[2] == compressor.target_boxsize[2])
                        compressor.reached_target = true
                    end
                end
            end
        end
    end
end

@inline function get_new_acceptance_rates(sim::ColloidSim, tuner::MoveSizeTuner)
    acc_trans = sim.accepted_translations - tuner.prev_accepted_translations
    rej_trans = sim.rejected_translations - tuner.prev_rejected_translations
    acc_rot = sim.accepted_rotations - tuner.prev_accepted_rotations
    rej_rot = sim.rejected_rotations - tuner.prev_rejected_rotations
    return acc_trans / (acc_trans + rej_trans), acc_rot / (acc_rot + rej_rot)
end

@inline function set_tuner_flags!(
        tuner::MoveSizeTuner, translation_acceptance::Real, rotation_acceptance::Real)
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

@inline function set_tuner_prev_values!(sim::ColloidSim, tuner::MoveSizeTuner)
    tuner.prev_accepted_translations = sim.accepted_translations
    tuner.prev_rejected_translations = sim.rejected_translations
    tuner.prev_accepted_rotations = sim.accepted_rotations
    tuner.prev_rejected_rotations = sim.rejected_rotations
end

@inline function get_force_compress_dims(sim::ColloidSim, compressor::ForcefulCompressor)
    CUDA.allowscalar() do
        scale_factor = max(compressor.minscale,
            1.0 - sim.move_radius / (2 * sim.colloid.radius))

        if sim.colloid.boxsize[1] < compressor.target_boxsize[1]
            lxnew = min(sim.colloid.boxsize[1] / scale_factor,
                        compressor.target_boxsize[1])
        else
            lxnew = max(sim.colloid.boxsize[1] * scale_factor,
                        compressor.target_boxsize[1])
        end

        if sim.colloid.boxsize[2] < compressor.target_boxsize[2]
            lynew = min(sim.colloid.boxsize[2] / scale_factor,
                        compressor.target_boxsize[2])
        else
            lynew = max(sim.colloid.boxsize[2] * scale_factor,
                        compressor.target_boxsize[2])
        end

        return lxnew, lynew
    end
end

@inline function needs_compression(sim::ColloidSim, compressor::ForcefulCompressor)
    return (!compressor.completed && compressor.cond(sim.timestep)
        && !has_overlap(sim.colloid, sim.cell_list)
        && ((!sim.gpu && !has_violation(sim.colloid, sim.constraints))
            || (sim.gpu && count_violations_gpu(sim.colloid, sim.constraints) == 0)))
end

@inline function set_complete_flag!(sim::ColloidSim, compressor::ForcefulCompressor)
    if sim.gpu && iszero(count_overlaps(sim.colloid, sim.cell_list)
                         + count_violations_gpu(sim.colloid, sim.constraints))
        sim.cell_list = CuCellList(sim.colloid, sim.cell_list.shift)
        compressor.completed = true
    elseif iszero(count_overlaps(sim.colloid, sim.cell_list)
                  + count_violations(sim.colloid, sim.constraints))
        compressor.completed = true
    end
end

@inline function apply_compression!(sim::ColloidSim, compressor::ForcefulCompressor)
    lxnew, lynew = get_force_compress_dims(sim, compressor)
    CUDA.@allowscalar lxold, lyold = sim.colloid.boxsize
    
    CUDA.@allowscalar sim.colloid.boxsize[1], sim.colloid.boxsize[2] = lxnew, lynew
    pos_scale = (lxnew / lxold, lynew / lyold)
    sim.colloid.centers .*= pos_scale
    return pos_scale, lxold, lyold
end
