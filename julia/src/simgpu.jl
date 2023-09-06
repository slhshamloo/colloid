function apply_step!(sim::Simulation, cell_list::CuCellList, constraints::RawConstraints)
    blockthreads = numthreads[1] * numthreads[2]
    sweeps = ceil(Int, mean(cell_list.counts))

    randnums = CUDA.rand(sim.numtype, 4, size(cell_list.counts, 1),
                         size(cell_list.counts, 2), sweeps)
    randchoices = CUDA.rand(Bool, size(cell_list.cells, 2),
                            size(cell_list.cells, 3), sweeps)
    accept = CuArray(zeros(Int, 4))

    for sweep in 1:sweeps
        maxcount = maximum(cell_list.counts)
        groupcount = 9 * maxcount + length(sim.constraints)
        groups_per_block = blockthreads ÷ groupcount
        for color in shuffle(1:4)
            cellcount = getcellcount(cell_list, color)
            numblocks = cellcount ÷ groups_per_block + 1
            @cuda(threads=blockthreads, blocks=numblocks,
                  shmem = groups_per_block * (2 * sizeof(Int32) + sizeof(sim.numtype)),
                  apply_parallel_step!(sim.colloid, cell_list, color, sweep, maxcount,
                      groupcount, cellcount, sim.move_radius, sim.rotation_span, sim.beta,
                      randnums, randchoices, accept, constraints, sim.potential,
                      sim.pairpotential))
        end
        direction = ((1, 0), (-1, 0), (0, 1), (0, -1))[rand(1:4)]
        shift = (direction[2] == 0 ? cell_list.width[1] : cell_list.width[2]) * (
            rand(sim.numtype) / 2)
        shift_cells!(sim.colloid, cell_list, direction, shift)
    end
    count_accepted_and_rejected_moves!(sim, accept)
end

@inline getcellcount(cell_list, color) = (
    (size(cell_list.cells, 2) ÷ 2 + (color % 2) * (size(cell_list.cells, 2) % 2))
    * (size(cell_list.cells, 3) ÷ 2 + (color ÷ 3) * (size(cell_list.cells, 3) % 2)))

function count_accepted_and_rejected_moves!(sim::Simulation, accept::CuArray)
    accept = Vector(accept)
    sim.accepted_translations += accept[1]
    sim.rejected_translations += accept[2]
    sim.accepted_rotations += accept[3]
    sim.rejected_rotations += accept[4]
end

function apply_parallel_step!(colloid::Colloid, cell_list::CuCellList, color::Integer,
        sweep::Integer, maxcount::Integer, groupcount::Integer, cellcount::Integer,
        move_radius::Real, rotation_span::Real, beta::Real, randnums::CuDeviceArray,
        randchoices::CuDeviceArray, accept::CuDeviceArray, constraints::RawConstraints,
        potential::Union{Function, Nothing}, pairpotential::Union{Function, Nothing})
    is_thread_active = true
    groups_per_block = blockDim().x ÷ groupcount
    shared_memory = CuDynamicSharedArray(Int32, 2 * groups_per_block)
    reject_count = @view shared_memory[1:groups_per_block]
    idx = @view shared_memory[groups_per_block+1:end]
    group_potentials = CuDynamicSharedArray(eltype(colloid.centers), groups_per_block,
                                            2 * groups_per_block * sizeof(Int32))

    if threadIdx().x > groups_per_block * groupcount
        is_thread_active = false
        group, thread, i, j = 0, 0, 0, 0
    else
        group, thread = divrem(threadIdx().x - 1, groupcount)
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

    apply_parallel_move!(colloid, cell_list, randnums, randchoices, accept, constraints,
        potential, pairpotential, idx, reject_count, group_potentials, move_radius,
        rotation_span, beta, groupcount, maxcount, group, thread, sweep,
        i, j, is_thread_active)
    return
end

function apply_parallel_move!(colloid::Colloid, cell_list::CuCellList,
        randnums::CuDeviceArray, randchoices::CuDeviceArray, accept::CuDeviceArray,
        constraints::RawConstraints, potential::Union{Function, Nothing},
        pairpotential::Union{Function, Nothing}, idx::SubArray, reject_count::SubArray,
        group_potentials::CuDeviceArray, move_radius::Real, rotation_span::Real, beta::Real,
        groupcount::Integer, maxcount::Integer, group::Integer, thread::Integer,
        sweep::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    xprev, yprev, angle_change = apply_parallel_trial!(
        colloid, cell_list, randnums, randchoices, group_potentials, idx, reject_count,
        move_radius, rotation_span, group, thread, sweep, i, j, is_thread_active)
    CUDA.sync_threads()

    if isnothing(pairpotential)
        checkoverlap!(colloid, cell_list, constraints, idx, reject_count,
                      maxcount, group, thread, i, j, is_thread_active)
    elseif groupcount > 9 * maxcount
        checkconstraint!(colloid, constraints, idx, reject_count,
                         maxcount, group, thread, is_thread_active)
    end
    if (!isnothing(potential) || !isnothing(pairpotential))
        checkpotential!(colloid, cell_list, xprev, yprev, angle_change, beta, potential,
                        pairpotential, randnums, group_potentials, idx, reject_count,
                        maxcount, group, thread, sweep, i, j, is_thread_active)
    end
    CUDA.sync_threads()

    if is_thread_active && thread == 0
        apply_parallel_acceptance!(colloid, accept, xprev, yprev, angle_change, idx[group],
                                   randchoices[i, j, sweep], reject_count[group] > 0)
    end
    return
end

function apply_parallel_trial!(colloid::Colloid, cell_list::CuCellList,
        randnums::CuDeviceArray, randchoices::CuDeviceArray,
        group_potentials::CuDeviceArray, idx::SubArray, reject_count::SubArray,
        move_radius::Real, rotation_span::Real, group::Integer, thread::Integer,
        sweep::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    xprev, yprev = zero(eltype(colloid.centers)), zero(eltype(colloid.centers))
    angle_change = zero(typeof(rotation_span))
    if is_thread_active && thread == 0
        group_potentials[group] = zero(eltype(group_potentials))
        idx[group] = cell_list.cells[ceil(Int, randnums[1, i, j, sweep]
                                     * cell_list.counts[i, j]), i, j]
        if randchoices[i, j, sweep]
            xprev = colloid.centers[1, idx[group]]
            yprev = colloid.centers[2, idx[group]]
            r = move_radius * randnums[2, i, j, sweep]
            θ = 2π * randnums[3, i, j, sweep]
            x, y = r * cos(θ), r * sin(θ)
            move!(colloid, idx[group], x, y)
            reject_count[group] = ((i, j) != get_cell_list_indices(
                colloid, cell_list, idx[group]))
        else
            angle_change = rotation_span * (randnums[2, i, j, sweep] - 0.5)
            colloid.angles[idx[group]] += angle_change
            reject_count[group] = 0
        end
    end
    return xprev, yprev, angle_change
end

function checkoverlap!(colloid::Colloid, cell_list::CuCellList, constraints::RawConstraints,
        idx::SubArray, reject_count::SubArray, maxcount::Integer, group::Integer,
        thread::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    passed_circumference_check, passed_fast_check = false, false
    if is_thread_active && reject_count[group] == 0
        constraint_index = thread - 9 * maxcount + 1
        if constraint_index > 0
            passed_fast_check, dist, distnorm = constraint_step_one!(
                colloid, constraints, reject_count, idx[group], constraint_index, group)
        else
            passed_circumference_check, centerangle, neighbor, dist, distnorm = 
                overlap_step_one!(colloid, cell_list, reject_count, idx[group],
                                  group, thread, maxcount, i, j)
        end
    end
    CUDA.sync_threads()

    if is_thread_active && reject_count[group] == 0
        if passed_circumference_check
            if (_is_vertex_overlapping(colloid, idx[group], neighbor, distnorm, centerangle)
                    || _is_vertex_overlapping(colloid, neighbor, idx[group], distnorm,
                                              π + centerangle))
                CUDA.@atomic reject_count[group] += 1
            end
        elseif passed_fast_check
            if slowcheck(colloid, idx[group], dist, distnorm, constraints, constraint_index)
                CUDA.@atomic reject_count[group] += 1
            end
        end
    end
end

function checkconstraint!(colloid::Colloid, constraints::RawConstraints,
        idx::SubArray, reject_count::SubArray, maxcount::Integer, group::Integer,
        thread::Integer, is_thread_active::Bool)
    passed_fast_check = false
    if is_thread_active && reject_count[group] == 0
        constraint_index = thread - 9 * maxcount + 1
        if constraint_index > 0
            passed_fast_check, dist, distnorm = constraint_step_one!(
                colloid, constraints, reject_count, idx[group], constraint_index, group)
        end
    end
    CUDA.sync_threads()

    if is_thread_active && reject_count[group] == 0
        if passed_fast_check
            if slowcheck(colloid, idx[group], dist, distnorm, constraints, constraint_index)
                CUDA.@atomic reject_count[group] += 1
            end
        end
    end
end
 
function checkpotential!(colloid::Colloid, cell_list::CuCellList, xprev::Real, yprev::Real,
        angle_change::Real, beta::Real, potential::Union{Function, Nothing},
        pairpotential::Union{Function, Nothing}, randnums::CuDeviceArray,
        group_potentials::CuDeviceArray, idx::SubArray, reject_count::SubArray,
        maxcount::Integer, group::Integer, thread::Integer,
        sweep::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    if is_thread_active && reject_count[group] == 0 && thread <= 9 * maxcount
        if !isnothing(pairpotential)
            ineighbor, jneighbor, k = get_neighbor_indices(
                cell_list, thread, maxcount, i, j)
            if k <= cell_list.counts[ineighbor, jneighbor]
                neighbor = cell_list.cells[k, ineighbor, jneighbor]
                if idx[group] != neighbor
                    CUDA.@atomic group_potentials[group] += pairpotential(
                        colloid, idx[group], neighbor, xprev, yprev, angle_change)
                end
            end
        end
        if !isnothing(potential) && thread == 0
            CUDA.@atomic group_potentials[group] += potential(
                colloid, idx[group], xprev, yprev, angle_change)
        end
    end
    CUDA.sync_threads()

    if is_thread_active && reject_count[group] == 0 && thread == 0
        if randnums[4, i, j, sweep] > exp(-beta * group_potentials[group])
            reject_count[group] += 1
        end
    end
end

@inline function apply_parallel_acceptance!(colloid::Colloid, accept::CuDeviceArray,
        xprev::Real, yprev::Real, angle_change::Real, idx::Integer,
        is_translation::Bool, rejected::Bool)
    if rejected
        if is_translation
            colloid.centers[1, idx] = xprev
            colloid.centers[2, idx] = yprev
            CUDA.@atomic accept[2] += 1
        else
            colloid.angles[idx] -= angle_change
            CUDA.@atomic accept[4] += 1
        end
    else
        if is_translation
            CUDA.@atomic accept[1] += 1
        else
            CUDA.@atomic accept[3] += 1
        end
    end
end

@inline function get_neighbor_indices(cell_list::CuCellList, thread::Integer,
        maxcount::Integer, i::Integer, j::Integer)
    relpos, kneighbor = divrem(thread, maxcount)
    kneighbor += 1
    jdelta, idelta = divrem(relpos, 3)
    ineighbor = mod(i + idelta - 2, size(cell_list.cells, 2)) + 1
    jneighbor = mod(j + jdelta - 2, size(cell_list.cells, 3)) + 1
    return ineighbor, jneighbor, kneighbor
end

function overlap_step_one!(colloid::Colloid, cell_list::CuCellList,
        reject_count::SubArray, idx::Integer, group::Integer, thread::Integer,
        maxcount::Integer, i::Integer, j::Integer)
    passed_circumference_check, neighbor = false, zero(Int32)
    distnorm, centerangle = zero(eltype(colloid.centers)), zero(eltype(colloid.centers))
    dist = (zero(eltype(colloid.centers)), zero(eltype(colloid.centers)))

    ineighbor, jneighbor, k = get_neighbor_indices(cell_list, thread, maxcount, i, j)
    if k <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[k, ineighbor, jneighbor]
        if idx != neighbor
            definite_overlap, out_of_range, dist, distnorm = _overlap_range(
                colloid, idx, neighbor)
            if definite_overlap
                CUDA.@atomic reject_count[group] += 1
            elseif !out_of_range
                passed_circumference_check = true
                centerangle = (dist[2] < 0 ? -1 : 1) * acos(dist[1] / distnorm)
            end
        end
    end
    return passed_circumference_check, centerangle, neighbor, dist, distnorm
end

function constraint_step_one!(colloid::Colloid, constraints::RawConstraints,
        reject_count::SubArray, idx::Integer, constraint_index::Integer, group::Integer)
    passed_fast_check = false
    definite_overlap, out_of_range, dist, distnorm = fastcheck(
        colloid, idx, constraints, constraint_index)
    if definite_overlap
        CUDA.@atomic reject_count[group] += 1 
    elseif !out_of_range
        passed_fast_check = true
    end
    return passed_fast_check, dist, distnorm
end
