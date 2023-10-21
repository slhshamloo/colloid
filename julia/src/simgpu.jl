function apply_step_gpu!(sim::HPMCSimulation)
    sweeps = ceil(Int, mean(sim.cell_list.counts))
    randnums = CUDA.rand(sim.numtype, 4, size(sim.cell_list.cells, 2),
                         size(sim.cell_list.cells, 3), sweeps)
    randchoices = CUDA.rand(Bool, size(sim.cell_list.cells, 2),
                            size(sim.cell_list.cells, 3), sweeps)
    accept = CuArray(zeros(Int, 4))
    cusim = build_cusim(sim)

    for sweep in 1:sweeps
        maxcount = maximum(sim.cell_list.counts)
        groucount = 9 * maxcount + length(sim.constraints)
        groups_per_block = numthreads ÷ groucount
        for color in shuffle(1:4)
            cellcount = getcellcount(sim.cell_list, color)
            numblocks = cellcount ÷ groups_per_block + 1
            @cuda(threads=numthreads, blocks=numblocks,
                  shmem = groups_per_block * (sizeof(Int32) + sizeof(sim.numtype)),
                  apply_parallel_step!(cusim, sim.cell_list, randnums, randchoices, accept,
                                       color, sweep, maxcount, groucount, cellcount))
        end
        direction = ((1, 0), (-1, 0), (0, 1), (0, -1))[rand(1:4)]
        shift = (direction[2] == 0 ? sim.cell_list.width[1] : sim.cell_list.width[2]) * (
            rand(sim.numtype) / 2)
        shift_cells!(sim.particles, sim.cell_list, direction, shift)
    end
    count_accepted_and_rejected_moves!(sim, accept)
end

function build_cusim(sim::HPMCSimulation)
    constraints = build_raw_constraints(sim.constraints, sim.numtype)
    if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
        if length(sim.particle_potentials) == 0
            sim.particle_potentials = CuArray(
                zeros(sim.numtype, count(sim.particles)))
        else
            sim.particle_potentials .= 0
        end
        calculate_potentials!(sim.particles, sim.cell_list, sim.particle_potentials,
                              sim.potential, sim.pairpotential)
    end
    CuHPMCSimulation(sim.particles, constraints, sim.move_radius, sim.rotation_span,
                 sim.beta, sim.potential, sim.pairpotential, sim.particle_potentials)
end

@inline getcellcount(cell_list, color) = (
    (size(cell_list.cells, 2) ÷ 2 + (color % 2) * (size(cell_list.cells, 2) % 2))
    * (size(cell_list.cells, 3) ÷ 2 + (color ÷ 3) * (size(cell_list.cells, 3) % 2)))

function count_accepted_and_rejected_moves!(sim::HPMCSimulation, accept::CuArray)
    accept = Vector(accept)
    sim.accepted_translations += accept[1]
    sim.rejected_translations += accept[2]
    sim.accepted_rotations += accept[3]
    sim.rejected_rotations += accept[4]
end

function apply_parallel_step!(cusim::CuHPMCSimulation, cell_list::CuCellList, 
        randnums::CuDeviceArray, randchoices::CuDeviceArray, accept::CuDeviceArray,
        color::Integer, sweep::Integer, maxcount::Integer, groucount::Integer,
        cellcount::Integer)
    is_thread_active = true
    groups_per_block = blockDim().x ÷ groucount
    idx = CuDynamicSharedArray(Int32, groups_per_block)
    group_potentials = CuDynamicSharedArray(eltype(cusim.particles.centers),
        groups_per_block, groups_per_block * sizeof(Int32))

    if threadIdx().x > groups_per_block * groucount
        is_thread_active = false
        group, thread, i, j = 0, 0, 0, 0
    else
        group, thread = divrem(threadIdx().x - 1, groucount)
        group += 1
        cell = (blockIdx().x - 1) * groups_per_block + group
        if cell > cellcount
            is_thread_active = false
            i, j = 0, 0
        else
            j, i = divrem(cell - 1, size(cell_list.cells, 2) ÷ 2
                          + (color % 2) * (size(cell_list.cells, 2) % 2))
            i = 2 * (i + 1) - (color % 2)
            j = 2 * (j + 1) - (color ÷ 3)
            is_thread_active = (cell_list.counts[i, j] != 0)
        end
    end

    apply_parallel_move!(cusim, cell_list, randnums, randchoices, accept, idx,
        group_potentials, groucount, maxcount, group, thread,
        sweep, i, j, is_thread_active)
    return
end

function apply_parallel_move!(cusim::CuHPMCSimulation, cell_list::CuCellList,
        randnums::CuDeviceArray, randchoices::CuDeviceArray, accept::CuDeviceArray,
        idx::CuDeviceArray, group_potentials::CuDeviceArray, groucount::Integer,
        maxcount::Integer, group::Integer, thread::Integer, sweep::Integer,
        i::Integer, j::Integer, is_thread_active::Bool)
    if is_thread_active && thread == 0
        xprev, yprev, angle_change = apply_parallel_trial!(
            cusim, cell_list, randnums, randchoices, idx, group_potentials,
            group, sweep, i, j)
    end
    CUDA.sync_threads()

    if isnothing(cusim.pairpotential)
        checkoverlap!(cusim.particles, cell_list, cusim.constraints, idx, group_potentials,
                      maxcount, group, thread, i, j, is_thread_active)
        CUDA.sync_threads()
    elseif groucount > 9 * maxcount
        checkconstraint!(cusim.particles, cusim.constraints, idx, group_potentials,
                         maxcount, group, thread, is_thread_active)
        CUDA.sync_threads()
    end
    accepted = is_thread_active && iszero(group_potentials[group])
    if !isnothing(cusim.potential) || !isnothing(cusim.pairpotential)
        accepted = checkpotential!(cusim, cell_list, randnums, idx, group_potentials,
                                   maxcount, group, thread, sweep, i, j, accepted)
        CUDA.sync_threads()
    end

    if is_thread_active && thread == 0
        apply_parallel_acceptance!(cusim, accept, group_potentials, xprev, yprev,
            angle_change, idx[group], group, randchoices[i, j, sweep], accepted)
    end
    return
end

function apply_parallel_trial!(cusim::CuHPMCSimulation, cell_list::CuCellList,
        randnums::CuDeviceArray, randchoices::CuDeviceArray,
        idx::CuDeviceVector, group_potentials::CuDeviceArray,
        group::Integer, sweep::Integer, i::Integer, j::Integer)
    xprev, yprev = zero(eltype(cusim.particles.centers)), zero(eltype(cusim.particles.centers))
    angle_change = zero(eltype(cusim.particles.angles))
    idx[group] = cell_list.cells[
        ceil(Int, randnums[1, i, j, sweep] * cell_list.counts[i, j]), i, j]
    if randchoices[i, j, sweep]
        xprev = cusim.particles.centers[1, idx[group]]
        yprev = cusim.particles.centers[2, idx[group]]
        r = cusim.move_radius * randnums[2, i, j, sweep]
        θ = 2π * randnums[3, i, j, sweep]
        x, y = r * cos(θ), r * sin(θ)
        move!(cusim.particles, idx[group], x, y)
        group_potentials[group] = ((i, j) != get_cell_list_indices(
            cusim.particles, cell_list, idx[group]))
    else
        angle_change = convert(eltype(cusim.particles.angles),
            cusim.rotation_span * (randnums[2, i, j, sweep] - 0.5))
        cusim.particles.angles[idx[group]] += angle_change
        group_potentials[group] = zero(eltype(group_potentials))
    end
    return xprev, yprev, angle_change
end

function checkoverlap!(particles::RegularPolygons, cell_list::CuCellList, constraints::RawConstraints,
        idx::CuDeviceArray, group_potentials::CuDeviceArray, maxcount::Integer,
        group::Integer, thread::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    passed_circumference_check, passed_fast_check = false, false
    if is_thread_active && iszero(group_potentials[group])
        constraint_index = thread - 9 * maxcount + 1
        if constraint_index > 0
            passed_fast_check, dist, distnorm = constraint_step_one!(
                particles, constraints, group_potentials, idx[group], constraint_index, group)
        else
            passed_circumference_check, centerangle, neighbor, dist, distnorm = 
                overlap_step_one!(particles, cell_list, group_potentials, idx[group],
                                  group, thread, maxcount, i, j)
        end
    end
    CUDA.sync_threads()

    if is_thread_active && iszero(group_potentials[group])
        if passed_circumference_check
            if (_is_vertex_overlapping(particles, idx[group], neighbor, distnorm, centerangle)
                    || _is_vertex_overlapping(particles, neighbor, idx[group], distnorm,
                                              π + centerangle))
                CUDA.@atomic group_potentials[group] += one(eltype(group_potentials))
            end
        elseif passed_fast_check
            if slowcheck(particles, idx[group], dist, distnorm, constraints, constraint_index)
                CUDA.@atomic group_potentials[group] += one(eltype(group_potentials))
            end
        end
    end
end

function checkconstraint!(particles::RegularPolygons, constraints::RawConstraints,
        idx::CuDeviceArray, group_potentials::CuDeviceArray, maxcount::Integer,
        group::Integer, thread::Integer, is_thread_active::Bool)
    passed_fast_check = false
    if is_thread_active && group_potentials[group] == 0
        constraint_index = thread - 9 * maxcount + 1
        if constraint_index > 0
            passed_fast_check, dist, distnorm = constraint_step_one!(
                particles, constraints, group_potentials, idx[group], constraint_index, group)
        end
    end
    CUDA.sync_threads()

    if is_thread_active && group_potentials[group] == 0
        if passed_fast_check
            if slowcheck(particles, idx[group], dist, distnorm, constraints, constraint_index)
                CUDA.@atomic group_potentials[group] += one(eltype(group_potentials))
            end
        end
    end
end
 
function checkpotential!(cusim::CuHPMCSimulation, cell_list::CuCellList,
        randnums::CuDeviceArray, idx::CuDeviceArray, group_potentials::CuDeviceArray,
        maxcount::Integer, group::Integer, thread::Integer, sweep::Integer,
        i::Integer, j::Integer, passed_overlap_check::Bool)
    if passed_overlap_check && passed_overlap_check && thread <= 9 * maxcount
        if !isnothing(cusim.pairpotential)
            ineighbor, jneighbor, k = get_neighbor_indices(
                cell_list, thread, maxcount, i, j)
            if k <= cell_list.counts[ineighbor, jneighbor]
                neighbor = cell_list.cells[k, ineighbor, jneighbor]
                if idx[group] != neighbor
                    CUDA.@atomic group_potentials[group] += cusim.pairpotential(
                        cusim.particles, idx[group], neighbor)
                elseif !isnothing(cusim.potential)
                    CUDA.@atomic group_potentials[group] += cusim.potential(
                        cusim.particles, idx[group])
                end
            end
        elseif !isnothing(cusim.potential) && thread == 0
            CUDA.@atomic group_potentials[group] += cusim.potential(
                cusim.particles, idx[group])
        end
    end
    CUDA.sync_threads()

    if passed_overlap_check && thread == 0
        return randnums[4, i, j, sweep] <= exp(-cusim.beta * (
            group_potentials[group] - cusim.particle_potentials[idx[group]]))
    else
        return passed_overlap_check
    end
end

@inline function apply_parallel_acceptance!(cusim::CuHPMCSimulation, accept::CuDeviceArray,
        group_potentials::CuDeviceArray, xprev::Real, yprev::Real, angle_change::Real,
        idx::Integer, group::Integer, is_translation::Bool, accepted::Bool)
    if accepted
        if !isnothing(cusim.potential) || !isnothing(cusim.pairpotential)
            cusim.particle_potentials[idx] = group_potentials[group]
        end
        if is_translation
            CUDA.@atomic accept[1] += 1
        else
            CUDA.@atomic accept[3] += 1
        end
    else
        if is_translation
            cusim.particles.centers[1, idx] = xprev
            cusim.particles.centers[2, idx] = yprev
            CUDA.@atomic accept[2] += 1
        else
            cusim.particles.angles[idx] -= angle_change
            CUDA.@atomic accept[4] += 1
        end
    end
end

function overlap_step_one!(particles::RegularPolygons, cell_list::CuCellList,
        group_potentials::CuDeviceArray, idx::Integer, group::Integer,
        thread::Integer, maxcount::Integer, i::Integer, j::Integer)
    passed_circumference_check, neighbor = false, zero(Int32)
    distnorm, centerangle = zero(eltype(particles.centers)), zero(eltype(particles.centers))
    dist = (zero(eltype(particles.centers)), zero(eltype(particles.centers)))

    ineighbor, jneighbor, k = get_neighbor_indices(cell_list, thread, maxcount, i, j)
    if k <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[k, ineighbor, jneighbor]
        if idx != neighbor
            definite_overlap, out_of_range, dist, distnorm = _overlap_range(
                particles, idx, neighbor)
            if definite_overlap
                CUDA.@atomic group_potentials[group] += one(eltype(group_potentials))
            elseif !out_of_range
                passed_circumference_check = true
                centerangle = (dist[2] < 0 ? -1 : 1) * acos(dist[1] / distnorm)
            end
        end
    end
    return passed_circumference_check, centerangle, neighbor, dist, distnorm
end

function constraint_step_one!(particles::RegularPolygons, constraints::RawConstraints,
        group_potentials::CuDeviceArray, idx::Integer, cidx::Integer, group::Integer)
    passed_fast_check = false
    definite_overlap, out_of_range, dist, distnorm = fastcheck(
        particles, idx, constraints, cidx)
    if definite_overlap
        CUDA.@atomic group_potentials[group] += 1 
    elseif !out_of_range
        passed_fast_check = true
    end
    return passed_fast_check, dist, distnorm
end
