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
                    compressor.max_overlap_fraction * particlecount(sim.particles))
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

function update!(sim::HPMCSimulation, updater::AbstractBoxUpdater)
    if updater.cond(sim.timestep)
        if isnothing(sim.pairpotential)
            old_overlaps = count_overlaps(sim.particles, sim.cell_list)
        end
        if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
            oldpotential = sum(sim.particle_potentials)
        else
            oldpotential = 0.0
        end
        old_cell_list, params, violates = propose_box_update!(sim, updater)
        if violates || (isnothing(sim.pairpotential)
                        && count_overlaps(sim.particles, sim.cell_list) > old_overlaps)
            reject_box_move!(sim, updater, old_cell_list, params)
        else
            if rand() < get_metropolis_factor(
                    sim, updater, oldpotential, params.area_change)
                accept_box_move!(updater, params)
            else
                reject_box_move!(sim, updater, old_cell_list, params)
            end
        end
    end
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

function propose_box_update!(sim::HPMCSimulation, params::NamedTuple,
        get_update_param::Function, apply_update!::Function)
    old_cell_list = sim.cell_list
    update_param, return_params = get_update_param(sim, params)
    if sim.gpu
        apply_update!(sim, update_param)
        sim.cell_list = CuCellList(sim.particles, sim.cell_list.shift)
        violates = count_violations_gpu(sim.particles, sim.constraints) > 0
    else
        apply_update!(sim, update_param)
        sim.cell_list = SeqCellList(sim.particles)
        violates = has_violation(sim.particles, sim.constraints)
    end
    return old_cell_list, return_params, violates
end

function propose_box_update!(sim::HPMCSimulation, updater::BoxMover)
    CUDA.@allowscalar lxold, lyold = sim.particles.boxsize
    choice = get_choice(updater)
    if choice < 3
        if choice == 1
            lxnew, lynew = lxold + 2 * (rand() - 0.5) * updater.change[1], lyold
            CUDA.@allowscalar sim.particles.boxsize[1] = lxnew
        else
            lxnew, lynew = lxold, lyold + 2 * (rand() - 0.5) * updater.change[2]
            CUDA.@allowscalar sim.particles.boxsize[2] = lynew
        end
        params = (lxold=lxold, lyold=lyold, lxnew=lxnew, lynew=lynew, choice=choice)
        get_update_param, apply_update! = get_scale, apply_area_update!
    else
        shearchange = 2 * (rand() - 0.5) * updater.change[3]
        CUDA.@allowscalar sim.particles.boxshear[] += shearchange
        params = (shearchange=shearchange, choice=choice)
        get_update_param, apply_update! = get_shear, apply_shear_update!
    end
    return propose_box_update!(sim, params, get_update_param, apply_update!)
end

function propose_box_update!(sim::HPMCSimulation, updater::AreaUpdater)
    CUDA.@allowscalar lxold, lyold = sim.particles.boxsize
    area_new = lxold * lyold + 2 * (rand() - 0.5) * updater.area_change
    lxnew = √(lxold / lyold * area_new)
    lynew = lyold / lxold * lxnew
    CUDA.@allowscalar sim.particles.boxsize[1], sim.particles.boxsize[2] = lxnew, lynew
    return propose_box_update!(sim, (lxold=lxold, lyold=lyold, lxnew=lxnew, lynew=lynew),
                               get_scale, apply_area_update_isotropic!)
end

@inline function get_choice(updater::BoxMover)
    randnum = rand()
    for i in 1:3
        if randnum < updater.weights[i]
            return i
        end
        randnum -= updater.weights[i]
    end
    return 3
end

@inline function get_scale(sim, params)
    if sim.gpu
        scale = CuArray([params.lxnew / params.lxold, params.lynew / params.lyold])
    else
        scale = (params.lxnew / params.lxold, params.lynew / params.lyold)
    end
    return scale, (scale=scale, area_change = (
        params.lxnew * params.lynew - params.lxold * params.lyold), params...)
end

@inline function get_shear(sim, params)
    return params.shearchange, (area_change=0.0, params...)
end

@inline function apply_area_update_isotropic!(sim::HPMCSimulation, scale::AbstractVector)
    sim.particles.centers .*= scale
end

@inline function apply_area_update!(sim::HPMCSimulation, scale::AbstractVector)
    CUDA.@allowscalar shear = sim.particles.boxshear[]
    sim.particles.centers[1, :] .-= sim.particles.centers[2, :] * shear
    sim.particles.centers .*= scale
    sim.particles.centers[1, :] .+= sim.particles.centers[2, :] * shear
end

@inline function apply_shear_update!(sim::HPMCSimulation, shearchange::Real)
    sim.particles.centers[:, 1] .+= sim.particles.centers[:, 2] * shearchange
end

function get_metropolis_factor(
        sim::HPMCSimulation, updater::AbstractBoxUpdater,
        oldpotential::Real, area_change::Real)
    metropolis_exponent = sim.beta * updater.pressure * area_change
    if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
        calculate_potentials!(sim.particles, sim.cell_list,
            sim.particle_potentials, sim.potential, sim.pairpotential)
        newpotential = sum(sim.particle_potentials)
        metropolis_exponent += sim.beta * (newpotential - oldpotential)
    end
    return exp(-metropolis_exponent)
end

@inline function accept_box_move!(updater::AreaUpdater, params::NamedTuple)
    updater.accepted_moves += 1
end

@inline function accept_box_move!(updater::BoxMover, params::NamedTuple)
    updater.accepted_moves[params.choice] += 1
end

@inline function reject_box_move!(sim::HPMCSimulation, updater::AreaUpdater,
        old_cell_list::CellList, params::NamedTuple)
    sim.cell_list = old_cell_list
    sim.particles.centers ./= params.scale
    CUDA.@allowscalar sim.particles.boxsize[1], sim.particles.boxsize[2] = (
        params.lxold, params.lyold)
    updater.rejected_moves += 1
end

@inline function reject_box_move!(sim::HPMCSimulation, updater::BoxMover,
        old_cell_list::CellList, params::NamedTuple)
    sim.cell_list = old_cell_list
    if params.choice < 3
        reverse_area_update!(sim, params.scale)
        CUDA.@allowscalar sim.particles.boxsize[1], sim.particles.boxsize[2] = (
            params.lxold, params.lyold)
    else
        sim.particles.centers[:, 1] .-= sim.particles.centers[:, 2] * params.shearchange
        CUDA.@allowscalar sim.particles.boxshear[] -= params.shearchange
    end
    updater.rejected_moves[params.choice] += 1
end

@inline function reverse_area_update!(sim::HPMCSimulation, scale::AbstractVector)
    CUDA.@allowscalar shear = sim.particles.boxshear[]
    sim.particles.centers[1, :] .-= sim.particles.centers[2, :] * shear
    sim.particles.centers ./= scale
    sim.particles.centers[1, :] .+= sim.particles.centers[2, :] * shear
end
