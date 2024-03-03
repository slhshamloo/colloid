function katic_order(particles::RegularPolygons, k::Integer,
        neighbors::Integer = k; numtype::DataType = Float32)
    if isa(particles.centers, CuArray)
        return local_order(particles, CuCellList(particles), k,
            (neighbor_r, neighbor_angle) -> exp(1im * k * neighbor_angle) / neighbors,
            numtype=numtype)
    else
        return local_order(particles, SeqCellList(particles), k,
            (neighbor_r, neighbor_angle) -> sum(exp.(1im * k * neighbor_angle)) / neighbors,
            numtype=numtype)
    end
end

@inline katic_order(particles::RegularPolygons, cell_list::SeqCellList, k::Integer,
    neighbors::Integer = k; numtype::DataType = Float32) = local_order(
        particles, cell_list, k, (neighbor_r, neighbor_angle) -> sum(exp.(
            1im * k * neighbor_angle)) / neighbors, numtype=numtype)

@inline katic_order(particles::RegularPolygons, cell_list::CuCellList, k::Integer,
    neighbors::Integer = k; numtype::DataType = Float32) = local_order(
        particles, cell_list, neighbors, (neighbor_r, neighbor_angle) -> exp(
            1im * k * neighbor_angle) / neighbors, numtype=numtype)

function local_order(
        particles::RegularPolygons, cell_list::SeqCellList, neighbors::Integer,
        orderfunc::Function; iscomplex::Bool = true, numtype::DataType = Float32)
    orders = (iscomplex ? zeros(Complex{numtype}, particlecount(particles))
                        : zeros(numtype, particlecount(particles)))
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for idx in cell_list.cells[i, j]
            neighbor_r = Vector{numtype}(undef, 0)
            neighbor_angle = Vector{numtype}(undef, 0)
            for neighbor_cell in get_neighbors(cell_list.cells, i, j)
                for neighbor in neighbor_cell
                    r, angle = get_dist_and_angle(particles, idx, neighbor)
                    push!(neighbor_r, r)
                    push!(neighbor_angle, angle)
                end
            end
            if neighbors > 0
                if length(neighbor_r) >= neighbors
                    local_indices = partialsortperm(neighbor_r, 1:neighbors)
                    neighbor_r = neighbor_r[local_indices]
                    neighbor_angle = neighbor_angle[local_indices]
                    orders[idx] += orderfunc(neighbors, neighbor_r, neighbor_angle)
                end
            else
                if length(neighbor_r) > 0
                    orders[idx] += orderfunc(neighbor_r, neighbor_angle)
                end
            end
        end
    end
    return orders
end

function local_order(
        particles::RegularPolygons, cell_list::CuCellList, neighbors::Integer,
        orderfunc::Function; iscomplex::Bool = true, numtype::DataType = Float32)
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = particlecount(particles) ÷ groups_per_block + 1
    orders = (iscomplex ? zeros(Complex{numtype}, particlecount(particles))
                        : zeros(numtype, particlecount(particles)))
    orders = CuArray(orders)
    if neighbors > 0
        @cuda(threads=numthreads, blocks=numblocks,
              shmem = 2 * groups_per_block * (groupcount + neighbors) * sizeof(numtype),
              local_order_parallel!(particles, cell_list, orders, orderfunc, neighbors,
                                    maxcount, groupcount, groups_per_block, numtype))
    else
        @cuda(threads=numthreads, blocks=numblocks,
              local_order_parallel!(particles, cell_list, orders, orderfunc,
                                    maxcount, groupcount, groups_per_block))
    end
    return Vector(orders)
end

function local_order_parallel!(particles::RegularPolygons, cell_list::CuCellList,
        orders::CuDeviceVector, orderfunc::Function, neighbors::Integer, maxcount::Integer,
        groupcount::Integer, groups_per_block::Integer, numtype::DataType)
    active_threads = groups_per_block * groupcount
    group_r, group_angle, neighbor_r, neighbor_angle = get_local_order_shmem(
        active_threads, groups_per_block, groupcount, neighbors, numtype)

    is_thread_active = threadIdx().x <= active_threads
    group, thread = divrem(threadIdx().x - 1, groupcount)
    group += 1
    if is_thread_active
        particle = (blockIdx().x - 1) * groups_per_block + group
        is_thread_active = particle <= particlecount(particles)
        if is_thread_active
            i, j = get_cell_list_indices(particles, cell_list, particle)
            is_thread_active = count_neighbors(cell_list.counts, i, j) >= neighbors
            if is_thread_active
                calc_neighbor!(particles, cell_list, group_r, group_angle,
                               particle, i, j, maxcount, thread)
            end
        end
    end
    CUDA.sync_threads()

    k_partition_select!(group_r, group_angle, neighbor_r, neighbor_angle, neighbors,
                        group, thread, groupcount, is_thread_active)
    if is_thread_active && thread == 0
        for iteridx in (group - 1) * neighbors + 1 : group * neighbors
            orders[particle] += orderfunc(neighbor_r[iteridx], neighbor_angle[iteridx])
        end
    end
    return
end

function local_order_parallel!(particles::RegularPolygons, cell_list::CuCellList,
        orders::CuDeviceVector, orderfunc::Function, maxcount::Integer,
        groupcount::Integer, groups_per_block::Integer)
    if threadIdx().x <= groups_per_block * groupcount
        group, thread = divrem(threadIdx().x - 1, groupcount)
        group += 1
        particle = (blockIdx().x - 1) * groups_per_block + group
        if particle <= particlecount(particles)
            i, j = get_cell_list_indices(particles, cell_list, particle)
            if count_neighbors(cell_list.counts, i, j) > 0
                ineighbor, jneighbor, kneighbor = get_neighbor_indices(
                    cell_list, thread, maxcount, i, j)
                if kneighbor <= cell_list.counts[ineighbor, jneighbor]
                    neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
                    if particle != neighbor
                        r, theta = get_dist_and_angle(particles, particle, neighbor)
                        CUDA.@atomic orders[particle] += orderfunc(r, theta)
                    end
                end
            end
        end
    end
    return
end

@inline function get_local_order_shmem(active_threads::Integer,
        groups_per_block::Integer, groupcount::Integer,
        neighbors::Integer, numtype::DataType)
    shared_memory = CuDynamicSharedArray(
        numtype, 2 * groups_per_block * (groupcount + neighbors))
    group_r = @view shared_memory[1:active_threads]
    group_angle = @view shared_memory[active_threads + 1 : 2 * active_threads]
    neighbor_r = @view shared_memory[
        2 * active_threads + 1 : 2 * active_threads + groups_per_block * neighbors]
    neighbor_angle = @view shared_memory[
        2 * active_threads + groups_per_block * neighbors + 1 : end]
    return group_r, group_angle, neighbor_r, neighbor_angle
end

function calc_neighbor!(particles::RegularPolygons, cell_list::CuCellList,
        group_r::SubArray, group_angle::SubArray, particle::Integer,
        i::Integer, j::Integer, maxcount::Integer, thread::Integer)
    ineighbor, jneighbor, kneighbor = get_neighbor_indices(
        cell_list, thread, maxcount, i, j)
    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        if particle != neighbor
            group_r[threadIdx().x], group_angle[threadIdx().x] = get_dist_and_angle(
                particles, particle, neighbor)
        else
            group_r[threadIdx().x] = typemax(eltype(group_r))
        end
    else
        group_r[threadIdx().x] = typemax(eltype(group_r))
    end
    return
end

function get_dist_and_angle(particles::RegularPolygons, i::Integer, j::Integer)
    rij = (particles.centers[1, i] - particles.centers[1, j],
           particles.centers[2, i] - particles.centers[2, j])
    rij = (rij[1] - rij[1] ÷ (particles.boxsize[1]/2) * particles.boxsize[1],
           rij[2] - rij[2] ÷ (particles.boxsize[2]/2) * particles.boxsize[2])
    r = sqrt(rij[1]^2 + rij[2]^2)
    angle = (rij[2] < 0 ? -1 : 1) * acos(rij[1] / r)
    return r, angle
end

function k_partition_select!(group_r::SubArray, group_angle::SubArray,
        neighbor_r::SubArray, neighbor_angle::SubArray, k::Integer,
        group::Integer, thread::Integer, groupcount::Integer, is_thread_active::Bool)
    if is_thread_active
        group_r = @view group_r[(group - 1) * groupcount + 1 : group * groupcount]
        group_angle = @view group_angle[(group - 1) * groupcount + 1 : group * groupcount]
        neighbor_r = @view neighbor_r[(group - 1) * k + 1 : group * k]
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
            neighbor_r[selection] = group_r[1]
            neighbor_angle[selection] = group_angle[1]
            group_r[1] = typemax(eltype(group_r))
        end
    end
    return
end

@inline function count_neighbors(counts::AbstractMatrix, i::Integer, j::Integer)
    (counts[mod(i - 2, size(counts, 1)) + 1, mod(j - 2, size(counts, 2)) + 1]
     + counts[mod(i - 2, size(counts, 1)) + 1, j]
     + counts[mod(i - 2, size(counts, 1)) + 1, mod(j, size(counts, 1)) + 1]
     + counts[i, mod(j - 2, size(counts, 1)) + 1]
     + counts[i, j] - 1
     + counts[i, mod(j, size(counts, 1)) + 1]
     + counts[mod(i, size(counts, 1)) + 1, mod(j - 2, size(counts, 2)) + 1]
     + counts[mod(i, size(counts, 1)) + 1, j]
     + counts[mod(i, size(counts, 1)) + 1, mod(j, size(counts, 1)) + 1])
end

@inline function get_neighbors(cells::AbstractMatrix, i::Integer, j::Integer)
    (cells[mod(i - 2, size(cells, 1)) + 1, mod(j - 2, size(cells, 2)) + 1],
     cells[mod(i - 2, size(cells, 1)) + 1, j],
     cells[mod(i - 2, size(cells, 1)) + 1, mod(j, size(cells, 1)) + 1],
     cells[i, mod(j - 2, size(cells, 1)) + 1],
     cells[i, j],
     cells[i, mod(j, size(cells, 1)) + 1],
     cells[mod(i, size(cell_list.counts, 1)) + 1, mod(j - 2, size(cells, 2)) + 1],
     cells[mod(i, size(cells, 1)) + 1, j],
     cells[mod(i, size(cells, 1)) + 1, mod(j, size(cells, 1)) + 1])
end
