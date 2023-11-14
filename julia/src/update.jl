function update!(sim::HPMCSimulation, tuner::MoveSizeTuner)
    if tuner.cond(sim.timestep)
        translation_acceptance, rotation_acceptance, npt_acceptance =
            get_new_acceptance_rates(sim, tuner)
        set_tuner_flags!(tuner, translation_acceptance, rotation_acceptance, npt_acceptance)

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

function update!(sim::HPMCSimulation, tuner::NPTTuner)
    if tuner.cond(sim.timestep)
        acc_moves = tuner.npt_mover.accepted_moves - tuner.prev_accepted_moves
        rej_moves = tuner.npt_mover.rejected_moves - tuner.prev_rejected_moves
        acceptance_rate = acc_moves / (acc_moves + rej_moves)
        if abs(acceptance_rate - tuner.target_acceptance_rate) <= tuner.tollerance
            if tuner.prev_tuned
                tuner.tuned = true
            else
                tuner.prev_tuned = true
            end
        end

        tuner.npt_mover.area_change = min(tuner.max_move_size,
            min(tuner.maxscale, (acceptance_rate + tuner.gamma)
                / (tuner.target_acceptance_rate + tuner.gamma))
                * tuner.npt_mover.area_change)

        tuner.prev_accepted_moves = tuner.npt_mover.accepted_moves
        tuner.prev_rejected_moves = tuner.npt_mover.rejected_moves
    end
end

function update!(sim::HPMCSimulation, compressor::ForcefulCompressor)
    if needs_compression(sim, compressor)
        if compressor.reached_target
            set_complete_flag!(sim, compressor)
        else
            pos_scale, lxnew, lynew, lxold, lyold = apply_compression!(sim, compressor)
            old_cell_list = sim.cell_list
            if sim.gpu
                sim.cell_list = CuCellList(sim.particles, sim.cell_list.shift,
                    maxwidth=minimum(pos_scale)*get_maxwidth(sim.particles))
                violations = count_violations_gpu(sim.particles, sim.constraints)
            else
                sim.cell_list = SeqCellList(sim.particles)
                violations = count_violations(sim.particles, sim.constraints)
            end
            if violations + count_overlaps(sim.particles, sim.cell_list) > (
                    compressor.max_overlap_fraction * count(sim.particles))
                CUDA.@allowscalar sim.particles.boxsize[1], sim.particles.boxsize[2] = (
                    lxold, lyold)
                sim.particles.centers ./= pos_scale
                sim.cell_list = old_cell_list
            else
                if (lxnew == compressor.target_boxsize[1]
                        && lynew == compressor.target_boxsize[2])
                    compressor.reached_target = true
                end
            end
        end
    end
end

function update!(sim::HPMCSimulation, npt::NPTMover)
    if npt.cond(sim.timestep)
        if isnothing(sim.pairpotential)
            old_overlaps = count_overlaps(sim.particles, sim.cell_list)
        end
        if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
            oldpotential = sum(sim.particle_potentials)
        end
        CUDA.@allowscalar lxold, lyold = sim.particles.boxsize
        old_area = lxold * lyold
        old_cell_list, scale, new_area, violates = propose_npt_move!(
            sim, npt, lxold, lyold, old_area)
        if violates || (isnothing(sim.pairpotential)
                        && count_overlaps(sim.particles, sim.cell_list) > old_overlaps)
            reject_npt_move!(sim, npt, old_cell_list, scale, lxold, lyold)
        else
            metropolis_factor = sim.beta * npt.pressure * (new_area - old_area)
            if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
                calculate_potentials!(sim.particles, sim.cell_list,
                    sim.particle_potentials, sim.potential, sim.pairpotential)
                newpotential = sum(sim.particle_potentials)
                metropolis_factor += sim.beta * (newpotential - oldpotential)
            end
            if rand() < exp(-metropolis_factor)
                npt.accepted_moves += 1
            else
                reject_npt_move!(sim, npt, old_cell_list, scale, lxold, lyold)
            end
        end
    end
end

@inline function get_new_acceptance_rates(sim::HPMCSimulation, tuner::MoveSizeTuner)
    acc_trans = sim.accepted_translations - tuner.prev_accepted_translations
    rej_trans = sim.rejected_translations - tuner.prev_rejected_translations
    acc_rot = sim.accepted_rotations - tuner.prev_accepted_rotations
    rej_rot = sim.rejected_rotations - tuner.prev_rejected_rotations
    return acc_trans / (acc_trans + rej_trans), acc_rot / (acc_rot + rej_rot)
end

@inline function set_tuner_flags!(tuner::MoveSizeTuner,
        translation_acceptance::Real, rotation_acceptance::Real, npt_acceptance::Real)
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

@inline function set_tuner_prev_values!(sim::HPMCSimulation, tuner::MoveSizeTuner)
    tuner.prev_accepted_translations = sim.accepted_translations
    tuner.prev_rejected_translations = sim.rejected_translations
    tuner.prev_accepted_rotations = sim.accepted_rotations
    tuner.prev_rejected_rotations = sim.rejected_rotations
end

@inline function get_force_compress_dims(
        sim::HPMCSimulation, compressor::ForcefulCompressor)
    CUDA.allowscalar() do
        scale_factor = max(compressor.minscale,
            1.0 - sim.move_radius / (2 * sim.particles.radius))

        if sim.particles.boxsize[1] < compressor.target_boxsize[1]
            lxnew = min(sim.particles.boxsize[1] / scale_factor,
                        compressor.target_boxsize[1])
        else
            lxnew = max(sim.particles.boxsize[1] * scale_factor,
                        compressor.target_boxsize[1])
        end

        if sim.particles.boxsize[2] < compressor.target_boxsize[2]
            lynew = min(sim.particles.boxsize[2] / scale_factor,
                        compressor.target_boxsize[2])
        else
            lynew = max(sim.particles.boxsize[2] * scale_factor,
                        compressor.target_boxsize[2])
        end

        return lxnew, lynew
    end
end

@inline function needs_compression(sim::HPMCSimulation, compressor::ForcefulCompressor)
    return (!compressor.completed && compressor.cond(sim.timestep)
        && !has_overlap(sim.particles, sim.cell_list)
        && ((!sim.gpu && !has_violation(sim.particles, sim.constraints))
            || (sim.gpu && count_violations_gpu(sim.particles, sim.constraints) == 0)))
end

@inline function set_complete_flag!(sim::HPMCSimulation, compressor::ForcefulCompressor)
    if sim.gpu && iszero(count_overlaps(sim.particles, sim.cell_list)
                         + count_violations_gpu(sim.particles, sim.constraints))
        sim.cell_list = CuCellList(sim.particles, sim.cell_list.shift)
        compressor.completed = true
    elseif iszero(count_overlaps(sim.particles, sim.cell_list)
                  + count_violations(sim.particles, sim.constraints))
        compressor.completed = true
    end
end

@inline function apply_compression!(sim::HPMCSimulation, compressor::ForcefulCompressor)
    lxnew, lynew = get_force_compress_dims(sim, compressor)
    CUDA.@allowscalar lxold, lyold = sim.particles.boxsize
    CUDA.@allowscalar sim.particles.boxsize[1], sim.particles.boxsize[2] = lxnew, lynew
    pos_scale = (lxnew / lxold, lynew / lyold)
    sim.particles.centers .*= pos_scale
    return pos_scale, lxnew, lynew, lxold, lyold
end

@inline function propose_npt_move!(
        sim::HPMCSimulation, npt::NPTMover, lxold::Real, lyold::Real, old_area::Real)
    new_area = old_area + 2 * (rand() - 0.5) * npt.area_change
    lxnew = √(lxold / lyold * new_area)
    lynew = lyold / lxold * lxnew
    CUDA.@allowscalar sim.particles.boxsize[1], sim.particles.boxsize[2] = lxnew, lynew
    old_cell_list = sim.cell_list
    scale = (lxnew / lxold, lynew / lyold)
    if sim.gpu
        sim.cell_list = CuCellList(sim.particles, sim.cell_list.shift,
            maxwidth=minimum(scale)*get_maxwidth(sim.particles))
        scale = CuArray([lxnew / lxold, lynew / lyold])
        sim.particles.centers .*= scale
        violates = count_violations_gpu(sim.particles, sim.constraints) > 0
    else
        sim.particles.centers .*= scale
        sim.cell_list = SeqCellList(sim.particles)
        violates = has_violation(sim.particles, sim.constraints)
    end
    return old_cell_list, scale, new_area, violates
end

@inline function reject_npt_move!(sim::HPMCSimulation, npt::NPTMover,
        old_cell_list::CellList, scale::Union{AbstractVector, Tuple{<:Real, <:Real}},
        lxold::Real, lyold::Real)
    sim.cell_list = old_cell_list
    sim.particles.centers ./= scale
    CUDA.@allowscalar sim.particles.boxsize[1], sim.particles.boxsize[2] = lxold, lyold
    npt.rejected_moves += 1
end
