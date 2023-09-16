function apply_step!(sim::ColloidSim, cell_list::CuCellList)
    blockthreads = numthreads[1] * numthreads[2]
    sweeps = ceil(Int, mean(cell_list.counts))

    randnums = CUDA.rand(sim.numtype, 4, size(cell_list.cells, 2),
                         size(cell_list.cells, 3), sweeps)
    randchoices = CUDA.rand(Bool, size(cell_list.cells, 2),
                            size(cell_list.cells, 3), sweeps)
    accept = CuArray(zeros(Int, 4))
    cusim = build_cusim(sim, cell_list)

    for sweep in 1:sweeps
        maxcount = maximum(cell_list.counts)
        groupcount = 9 * maxcount + length(sim.constraints)
        groups_per_block = blockthreads ÷ groupcount
        for color in shuffle(1:4)
            cellcount = getcellcount(cell_list, color)
            numblocks = cellcount ÷ groups_per_block + 1
            @cuda(threads=blockthreads, blocks=numblocks,
                  shmem = groups_per_block * (2 * sizeof(Int32) + sizeof(sim.numtype)),
                  apply_parallel_step!(cusim, cell_list, randnums, randchoices, accept,
                                       color, sweep, maxcount, groupcount, cellcount))
        end
        direction = ((1, 0), (-1, 0), (0, 1), (0, -1))[rand(1:4)]
        shift = (direction[2] == 0 ? cell_list.width[1] : cell_list.width[2]) * (
            rand(sim.numtype) / 2)
        shift_cells!(sim.colloid, cell_list, direction, shift)
    end
    count_accepted_and_rejected_moves!(sim, accept)
end

function build_cusim(sim::ColloidSim, cell_list::CuCellList)
    constraints = build_raw_constraints(sim.constraints, sim.numtype)
    if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
        if length(sim.particle_potentials) == 0
            sim.particle_potentials = CuArray(
                zeros(sim.numtype, particle_count(sim.colloid)))
        else
            sim.particle_potentials .= 0
        end
        calculate_potentials!(sim.colloid, cell_list, sim.particle_potentials,
                              sim.potential, sim.pairpotential)
    end
    CuColloidSim(sim.colloid, constraints, sim.move_radius, sim.rotation_span,
                 sim.beta, sim.potential, sim.pairpotential, sim.particle_potentials)
end

@inline getcellcount(cell_list, color) = (
    (size(cell_list.cells, 2) ÷ 2 + (color % 2) * (size(cell_list.cells, 2) % 2))
    * (size(cell_list.cells, 3) ÷ 2 + (color ÷ 3) * (size(cell_list.cells, 3) % 2)))

function count_accepted_and_rejected_moves!(sim::ColloidSim, accept::CuArray)
    accept = Vector(accept)
    sim.accepted_translations += accept[1]
    sim.rejected_translations += accept[2]
    sim.accepted_rotations += accept[3]
    sim.rejected_rotations += accept[4]
end

function apply_parallel_step!(cusim::CuColloidSim, cell_list::CuCellList, 
        randnums::CuDeviceArray, randchoices::CuDeviceArray, accept::CuDeviceArray,
        color::Integer, sweep::Integer, maxcount::Integer, groupcount::Integer,
        cellcount::Integer)
    is_thread_active = true
    groups_per_block = blockDim().x ÷ groupcount
    shared_memory = CuDynamicSharedArray(Int32, 2 * groups_per_block)
    reject_count = @view shared_memory[1:groups_per_block]
    idx = @view shared_memory[groups_per_block+1:end]
    group_potentials = CuDynamicSharedArray(eltype(cusim.colloid.centers), groups_per_block,
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

    apply_parallel_move!(cusim, cell_list, randnums, randchoices, accept, idx,
        reject_count, group_potentials, groupcount, maxcount, group, thread, sweep,
        i, j, is_thread_active)
    return
end

function apply_parallel_move!(cusim::CuColloidSim, cell_list::CuCellList,
        randnums::CuDeviceArray, randchoices::CuDeviceArray, accept::CuDeviceArray,
        idx::SubArray, reject_count::SubArray, group_potentials::CuDeviceArray,
        groupcount::Integer, maxcount::Integer, group::Integer, thread::Integer,
        sweep::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    xprev, yprev, angle_change = apply_parallel_trial!(
        cusim, cell_list, randnums, randchoices, group_potentials, idx, reject_count,
        group, thread, sweep, i, j, is_thread_active)
    CUDA.sync_threads()

    if isnothing(cusim.pairpotential)
        checkoverlap!(cusim.colloid, cell_list, cusim.constraints, idx, reject_count,
                      maxcount, group, thread, i, j, is_thread_active)
    elseif groupcount > 9 * maxcount
        checkconstraint!(cusim.colloid, cusim.constraints, idx, reject_count,
                         maxcount, group, thread, is_thread_active)
    end
    if (!isnothing(cusim.potential) || !isnothing(cusim.pairpotential))
        checkpotential!(cusim, cell_list, randnums, group_potentials, idx, reject_count,
                        maxcount, group, thread, sweep, i, j, is_thread_active)
    end
    CUDA.sync_threads()

    if is_thread_active && thread == 0
        apply_parallel_acceptance!(cusim, accept, group_potentials, xprev, yprev,
            angle_change, idx[group], group, randchoices[i, j, sweep],
            reject_count[group] > 0)
    end
    return
end

function apply_parallel_trial!(cusim::CuColloidSim, cell_list::CuCellList,
        randnums::CuDeviceArray, randchoices::CuDeviceArray,
        group_potentials::CuDeviceArray, idx::SubArray, reject_count::SubArray,
        group::Integer, thread::Integer, sweep::Integer, i::Integer, j::Integer,
        is_thread_active::Bool)
    xprev, yprev = zero(eltype(cusim.colloid.centers)), zero(eltype(cusim.move_radius))
    angle_change = zero(eltype(cusim.colloid.centers))
    if is_thread_active && thread == 0
        group_potentials[group] = zero(eltype(group_potentials))
        idx[group] = cell_list.cells[ceil(Int, randnums[1, i, j, sweep]
                                     * cell_list.counts[i, j]), i, j]
        if randchoices[i, j, sweep]
            xprev = cusim.colloid.centers[1, idx[group]]
            yprev = cusim.colloid.centers[2, idx[group]]
            r = cusim.move_radius * randnums[2, i, j, sweep]
            θ = 2π * randnums[3, i, j, sweep]
            x, y = r * cos(θ), r * sin(θ)
            move!(cusim.colloid, idx[group], x, y)
            reject_count[group] = ((i, j) != get_cell_list_indices(
                cusim.colloid, cell_list, idx[group]))
        else
            angle_change = cusim.rotation_span * (randnums[2, i, j, sweep] - 0.5)
            cusim.colloid.angles[idx[group]] += angle_change
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
 
function checkpotential!(cusim::CuColloidSim, cell_list::CuCellList,
        randnums::CuDeviceArray, group_potentials::CuDeviceArray, idx::SubArray,
        reject_count::SubArray, maxcount::Integer, group::Integer, thread::Integer,
        sweep::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    if is_thread_active && reject_count[group] == 0 && thread <= 9 * maxcount
        if !isnothing(cusim.pairpotential)
            ineighbor, jneighbor, k = get_neighbor_indices(
                cell_list, thread, maxcount, i, j)
            if k <= cell_list.counts[ineighbor, jneighbor]
                neighbor = cell_list.cells[k, ineighbor, jneighbor]
                if idx[group] != neighbor
                    CUDA.@atomic group_potentials[group] += cusim.pairpotential(
                        cusim.colloid, idx[group], neighbor)
                elseif !isnothing(cusim.potential)
                    CUDA.@atomic group_potentials[group] += cusim.potential(
                        cusim.colloid, idx[group])
                end
            end
        elseif !isnothing(cusim.potential) && thread == 0
            CUDA.@atomic group_potentials[group] += cusim.potential(
                cusim.colloid, idx[group])
        end
    end
    CUDA.sync_threads()

    if is_thread_active && reject_count[group] == 0 && thread == 0
        if randnums[4, i, j, sweep] > exp(-cusim.beta * (
                group_potentials[group] - cusim.particle_potentials[idx[group]]))
            reject_count[group] += 1
        end
    end
end

@inline function apply_parallel_acceptance!(cusim::CuColloidSim, accept::CuDeviceArray,
        group_potentials::CuDeviceArray, xprev::Real, yprev::Real, angle_change::Real,
        idx::Integer, group::Integer, is_translation::Bool, rejected::Bool)
    if rejected
        if is_translation
            cusim.colloid.centers[1, idx] = xprev
            cusim.colloid.centers[2, idx] = yprev
            CUDA.@atomic accept[2] += 1
        else
            cusim.colloid.angles[idx] -= angle_change
            CUDA.@atomic accept[4] += 1
        end
    else
        if !isnothing(cusim.potential) || !isnothing(cusim.pairpotential)
            cusim.particle_potentials[idx] = group_potentials[group]
        end
        if is_translation
            CUDA.@atomic accept[1] += 1
        else
            CUDA.@atomic accept[3] += 1
        end
    end
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
