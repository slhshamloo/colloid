function katic_order(colloid::Colloid, cell_list::SeqCellList, k::Integer;
                     numtype::DataType = Float32)
    orders = zeros(pcount(colloid), Complex{numtype})
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for idx in cell_list.cells[i, j]
            neighbor_r = Vector{numtype}(undef, 0)
            neighbor_angle = Vector{numtype}(undef, 0)
            for neighbor_cell in (
                    cell_list.cells[mod(i - 2, size(cell_list.counts, 1)) + 1,
                                    mod(j - 2, size(cell_list.counts, 2)) + 1],
                    cell_list.cells[mod(i - 2, size(cell_list.counts, 1)) + 1, j],
                    cell_list.cells[mod(i - 2, size(cell_list.counts, 1)) + 1,
                                    mod(j, size(cell_list.counts, 1)) + 1],
                    cell_list.cells[i, mod(j - 2, size(cell_list.counts, 1)) + 1],
                    cell_list.cells[i, j] - 1,
                    cell_list.cells[i, mod(j, size(cell_list.counts, 1)) + 1],
                    cell_list.cells[mod(i, size(cell_list.counts, 1)) + 1,
                                    mod(j - 2, size(cell_list.counts, 2)) + 1],
                    cell_list.cells[mod(i, size(cell_list.counts, 1)) + 1, j],
                    cell_list.cells[mod(i, size(cell_list.counts, 1)) + 1,
                                    mod(j, size(cell_list.counts, 1)) + 1])
                for neighbor in neighbor_cell
                    r, angle = get_dist_and_angle(colloid, idx, neighbor)
                    push!(neighbor_r, r)
                    push!(neighbor_angle, angle)
                end
            end
            if length(neighbor_r) >= k
                neighbor_angle = neighbor_angle[partialsortperm(neighbor_r, 1:k)]
                orders[idx] += sum(exp.(1im * k * neighbor_angle)) / k
            end
        end
    end
    return orders
end

function katic_order(colloid::Colloid, cell_list::CuCellList, k::Integer;
                     numtype::DataType = Float32)
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = pcount(colloid) ÷ groups_per_block + 1
    orders = CuArray(zeros(Complex{numtype}, pcount(colloid)))
    @cuda(threads=numthreads, blocks=numblocks,
          shmem = groups_per_block * (2 * groupcount + k) * sizeof(numtype),
          katic_order_parallel!(colloid, cell_list, orders, k, maxcount,
                                groupcount, groups_per_block, numtype))
    return Vector(orders)
end

function katic_order_parallel!(colloid::Colloid, cell_list::CuCellList,
        orders::CuDeviceVector, k::Integer, maxcount::Integer,
        groupcount::Integer, groups_per_block::Integer, numtype::DataType)
    active_threads = groups_per_block * groupcount
    shared_memory = CuDynamicSharedArray(numtype, groups_per_block * (2 * groupcount + k))
    group_r = @view shared_memory[1:active_threads]
    group_angle = @view shared_memory[
        active_threads+1:2*active_threads]
    neighbor_angle = @view shared_memory[2*active_threads+1:end]

    is_thread_active = threadIdx().x <= active_threads
    group, thread = divrem(threadIdx().x - 1, groupcount)
    group += 1
    if is_thread_active
        particle = (blockIdx().x - 1) * groups_per_block + group
        is_thread_active = particle <= pcount(colloid)
        if is_thread_active
            i, j = get_cell_list_indices(colloid, cell_list, particle)
            is_thread_active = count_neighbors(cell_list, i, j) >= k
            if is_thread_active
                calc_neighbor!(colloid, cell_list, group_r, group_angle,
                               particle, i, j, maxcount, thread)
            end
        end
    end
    CUDA.sync_threads()

    katic_partition_select!(group_r, group_angle, neighbor_angle, k,
        group, thread, groupcount, is_thread_active)

    if is_thread_active && thread == 0
        for iteridx in (group - 1) * k + 1 : group * k
            orders[particle] += exp(1im * k * neighbor_angle[iteridx]) / k
        end
    end
    return
end

function calc_neighbor!(colloid::Colloid, cell_list::CuCellList,
        group_r::SubArray, group_angle::SubArray, particle::Integer,
        i::Integer, j::Integer, maxcount::Integer, thread::Integer)
    ineighbor, jneighbor, kneighbor = get_neighbor_indices(
        cell_list, thread, maxcount, i, j)
    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        if particle != neighbor
            group_r[threadIdx().x], group_angle[threadIdx().x] = get_dist_and_angle(
                colloid, particle, neighbor)
        else
            group_r[threadIdx().x] = typemax(eltype(group_r))
        end
    else
        group_r[threadIdx().x] = typemax(eltype(group_r))
    end
    return
end

function get_dist_and_angle(colloid::Colloid, i::Integer, j::Integer)
    rij = (colloid.centers[1, i] - colloid.centers[1, j],
           colloid.centers[2, i] - colloid.centers[2, j])
    rij = (rij[1] - rij[1] ÷ (colloid.boxsize[1]/2) * colloid.boxsize[1],
           rij[2] - rij[2] ÷ (colloid.boxsize[2]/2) * colloid.boxsize[2])
    r = sqrt(rij[1]^2 + rij[2]^2)
    angle = (rij[2] < 0 ? -1 : 1) * acos(rij[1] / r)
    return r, angle
end

function katic_partition_select!(group_r::SubArray, group_angle::SubArray,
        neighbor_angle::SubArray, k::Integer, group::Integer, thread::Integer,
        groupcount::Integer, is_thread_active::Bool)
    if is_thread_active
        group_r = @view group_r[(group - 1) * groupcount + 1 : group * groupcount]
        group_angle = @view group_angle[(group - 1) * groupcount + 1 : group * groupcount]
        neighbor_angle = @view neighbor_angle[(group - 1) * k + 1 : group * k]
        thread += 1
    end
    for selection in 1:k
        step = 1
        while step < groupcount
            if is_thread_active && (thread - 1) % 2step == 0 && thread + step <= groupcount
                if group_r[thread] > group_r[thread + step]
                    group_r[thread], group_r[thread + step] = group_r[thread + step],
                        group_r[thread]
                    group_angle[thread], group_angle[thread + step] = (
                        group_angle[thread + step], group_angle[thread])
                end
            end
            step *= 2
            CUDA.sync_threads()
        end
        if is_thread_active && isone(thread)
            neighbor_angle[selection] = group_angle[1]
            group_r[1] = typemax(eltype(group_r))
        end
    end
    return
end

function count_neighbors(cell_list::CuCellList, i::Integer, j::Integer)
    return (
        cell_list.counts[mod(i - 2, size(cell_list.counts, 1)) + 1,
                         mod(j - 2, size(cell_list.counts, 2)) + 1]
        + cell_list.counts[mod(i - 2, size(cell_list.counts, 1)) + 1, j]
        + cell_list.counts[mod(i - 2, size(cell_list.counts, 1)) + 1,
                           mod(j, size(cell_list.counts, 1)) + 1]
        + cell_list.counts[i, mod(j - 2, size(cell_list.counts, 1)) + 1]
        + cell_list.counts[i, j] - 1
        + cell_list.counts[i, mod(j, size(cell_list.counts, 1)) + 1]
        + cell_list.counts[mod(i, size(cell_list.counts, 1)) + 1,
                         mod(j - 2, size(cell_list.counts, 2)) + 1]
        + cell_list.counts[mod(i, size(cell_list.counts, 1)) + 1, j]
        + cell_list.counts[mod(i, size(cell_list.counts, 1)) + 1,
                           mod(j, size(cell_list.counts, 1)) + 1]
    )
end
