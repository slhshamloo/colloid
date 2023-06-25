function apply_step!(sim::Simulation, cell_list::CuCellList)
    blockthreads = numthreads[1] * numthreads[2]
    sweeps = ceil(Int, mean(cell_list.counts))

    randnums = CUDA.rand(sim.numtype, 3, size(cell_list.counts, 1),
                         size(cell_list.counts, 2), sweeps)
    randchoices = CUDA.rand(Bool, size(cell_list.cells, 2),
                            size(cell_list.cells, 3), sweeps)

    accept = CuArray(trues(size(cell_list.cells, 2), size(cell_list.cells, 3), sweeps))
    constraints = build_raw_constraints(sim.constraints, eltype(sim.colloid.centers))

    for sweep in 1:sweeps
        maxcount = maximum(cell_list.counts)
        groupcount = 9 * maxcount + length(sim.constraints)
        groups_per_block = blockthreads ÷ groupcount
        for color in shuffle(1:4)
            cellcount = getcellcount(cell_list, color)
            numblocks = cellcount ÷ groups_per_block + 1
            @cuda(threads=blockthreads, blocks=numblocks,
                  shmem = 2 * groups_per_block * sizeof(Int32), apply_parallel_step!(
                  sim.colloid, cell_list, color, sweep, maxcount, groupcount, cellcount,
                  sim.move_radius, sim.rotation_span, randnums, randchoices, accept,
                  constraints))
        end
        direction = ((1, 0), (-1, 0), (0, 1), (0, -1))[rand(1:4)]
        shift = (direction[2] == 0 ? cell_list.width[1] : cell_list.width[2]) * (
            rand(sim.numtype) / 2)
        shift_cells!(sim.colloid, cell_list, direction, shift)
    end
    count_accepted_and_rejected_moves(sim, randchoices, accept)
end

@inline getcellcount(cell_list, color) = (
    (size(cell_list.cells, 2) ÷ 2 + (color % 2) * (size(cell_list.cells, 2) % 2))
    * (size(cell_list.cells, 3) ÷ 2 + (color ÷ 3) * (size(cell_list.cells, 3) % 2)))

function count_accepted_and_rejected_moves(
        sim::Simulation, randchoices::CuArray, accept::CuArray)
    translations = sum(randchoices)
    rotations = prod(size(randchoices))
    accepted_moves = sum(accept)
    accepted_translations = sum(accept .& randchoices)
    sim.accepted_translations += sum(accept .& randchoices)
    sim.rejected_translations += translations - accepted_translations
    accepted_rotations = accepted_moves - accepted_translations
    sim.accepted_rotations += accepted_rotations
    sim.rejected_rotations += rotations - accepted_rotations
end

function apply_parallel_step!(colloid::Colloid, cell_list::CuCellList,
        color::Integer, sweep::Integer, maxcount::Integer, groupcount::Integer,
        cellcount::Integer, move_radius::Real, rotation_span::Real,
        randnums::CuDeviceArray, randchoices::CuDeviceArray, accept::CuDeviceArray,
        constraints::RawConstraints)
    is_thread_active = true
    groups_per_block = blockDim().x ÷ groupcount
    shared_memory = CuDynamicSharedArray(Int32, 2 * groups_per_block)
    reject_count = @view shared_memory[1:groups_per_block]
    idx = @view shared_memory[groups_per_block+1:end]

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

    apply_parallel_move!(colloid, cell_list, randnums, randchoices,
        accept, constraints, reject_count, idx, move_radius, rotation_span,
        maxcount, group, thread, sweep, i, j, is_thread_active)
    return
end

function apply_parallel_move!(colloid::Colloid, cell_list::CuCellList,
        randnums::CuDeviceArray, randchoices::CuDeviceArray, accept::CuDeviceArray,
        constraints::RawConstraints, reject_count::SubArray, idx::SubArray,
        move_radius::Real, rotation_span::Real, maxcount::Integer, group::Integer,
        thread::Integer, sweep::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    if is_thread_active && thread == 0
        idx[group] = cell_list.cells[ceil(Int, randnums[1, i, j, sweep]
                                     * cell_list.counts[i, j]), i, j]
        if randchoices[i, j, sweep]
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
    CUDA.sync_threads()

    checkoverlap(colloid, cell_list, constraints, reject_count, idx,
                 maxcount, group, thread, i, j, is_thread_active)
    CUDA.sync_threads()

    if is_thread_active && thread == 0 && reject_count[group] > 0
        accept[i, j, sweep] = false
        if randchoices[i, j, sweep]
            move!(colloid, idx[group], -x, -y)
        else
            colloid.angles[idx[group]] -= angle_change
        end
    end
    return
end

function checkoverlap(colloid::Colloid, cell_list::CuCellList, constraints::RawConstraints,
        reject_count::SubArray, idx::SubArray, maxcount::Integer, group::Integer,
        thread::Integer, i::Integer, j::Integer, is_thread_active::Bool)
    passed_circumference_check, passed_fast_check = false, false
    if is_thread_active && reject_count[group] == 0
        constraint_index = thread - 9 * maxcount + 1
        if constraint_index > 0
            definite_overlap, out_of_range, dist, distnorm = fastcheck(
                colloid, idx[group], constraints, constraint_index)
            if definite_overlap
                CUDA.@atomic reject_count[group] += 1 
            elseif !out_of_range
                passed_fast_check = true
            end
        else
            relpos, k = divrem(thread, maxcount)
            k += 1
            jdelta, idelta = divrem(relpos, 3)
            ineighbor = mod(i + idelta - 2, size(cell_list.cells, 2)) + 1
            jneighbor = mod(j + jdelta - 2, size(cell_list.cells, 3)) + 1

            if k <= cell_list.counts[ineighbor, jneighbor]
                neighbor = cell_list.cells[k, ineighbor, jneighbor]
                if idx[group] != neighbor
                    definite_overlap, out_of_range, dist, distnorm = _overlap_range(
                        colloid, idx[group], neighbor)
                    if definite_overlap
                        CUDA.@atomic reject_count[group] += 1
                    elseif !out_of_range
                        passed_circumference_check = true
                        centerangle = (dist[2] < 0 ? -1 : 1) * acos(dist[1] / distnorm)
                    end
                end
            end
        end
    end
    CUDA.sync_threads()

    if is_thread_active && reject_count[group] == 0
        if passed_circumference_check && reject_count[group] == 0
            if (_is_vertex_overlapping(colloid, idx[group], neighbor, distnorm, centerangle)
                    || _is_vertex_overlapping(colloid, neighbor, idx[group], distnorm,
                                            π + centerangle))
                CUDA.@atomic reject_count[group] += 1
            end
        elseif passed_fast_check && reject_count[group] == 0
            if slowcheck(colloid, idx[group], dist, distnorm, constraints, constraint_index)
                CUDA.@atomic reject_count[group] += 1
            end
        end
    end
end
