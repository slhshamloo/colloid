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
    neighbors::Integer = k; numtype::DataType = Float32, returngpu::Bool = false
    ) = local_order(particles, cell_list, neighbors, (neighbor_r, neighbor_angle) -> exp(
            1im * k * neighbor_angle) / neighbors, numtype=numtype, returngpu=returngpu)

function solidliquid(particles::RegularPolygons, k::Integer, neighbors::Integer = k;
        threshold::Real = 0.7, rmax::Real = 2 * particles.radius)
    if isa(particles.centers, CuArray)
        cell_list = CuCellList(particles)
        katic_orders = katic_order(particles, cell_list, k, neighbors, returngpu=true)
        return solidliquid(
            particles, cell_list, katic_orders, threshold=threshold, rmax=rmax)
    else
        cell_list = SeqCellList(particles)
        katic_orders = katic_order(particles, cell_list, k, neighbors)
    end
end

@inline solidliquid(particles::RegularPolygons, cell_list::SeqCellList, k::Integer,
    neighbors::Integer = k; threshold::Real = 0.7, rmax::Real = 2 * particles.radius
    ) = solidliquid(particles, cell_list, katic_order(particles, cell_list, k, neighbors),
                    threshold=threshold, rmax=rmax)

@inline solidliquid(particles::RegularPolygons, cell_list::CuCellList, k::Integer,
    neighbors::Integer = k; threshold::Real = 0.7, rmax::Real = 2 * particles.radius
    ) = solidliquid(particles, cell_list,
                    katic_order(particles, cell_list, k, neighbors, returngpu=true),
                    threshold=threshold, rmax=rmax)

function pmft_angle_pair(particles::RegularPolygons;
        bins::Tuple{<:Integer, <:Integer} = (1024, 1024),
        rmax::Real = 2.5 * particles.radius)
    if isa(particles.centers, CuArray)
        return pmft_angle_pair(particles, CuCellList(particles), bins=bins, rmax=rmax)
    else
        return pmft_angle_pair(particles, SeqCellList(particles), bins=bins, rmax=rmax)
    end
end

function pmft_angle_pair(particles::RegularPolygons, cell_list::SeqCellList;
        bins::Tuple{<:Integer, <:Integer} = (1024, 1024),
        rmax::Real = 2.5 * particles.radius)
    angle_range = 2π / particles.sidenum
    binsize = angle_range ./ bins
    partition_function = zeros(Int32, bins)

    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for idx in cell_list.cells[i, j]
            for neighbor_cell in get_neighbors(cell_list.cells, i, j)
                for neighbor in neighbor_cell
                    r, theta = get_dist_and_angle(particles, idx, neighbor)
                    if r <= rmax
                        theta1 = mod(particles.angles[idx] - theta, angle_range)
                        theta2 = mod(particles.angles[neighbor] - theta + π, angle_range)
                        bin1 = Int(theta1 ÷ binsize[1]) + 1
                        bin2 = Int(theta2 ÷ binsize[2]) + 1
                        partition_function[bin1, bin2] += 1
                        partition_function[bin2, bin1] += 1
                    end
                end
            end
        end
    end

    return Matrix(log.(partition_function))
end

function pmft_angle_pair(particles::RegularPolygons, cell_list::CuCellList;
        bins::Tuple{<:Integer, <:Integer} = (1024, 1024),
        rmax::Real = 2.5 * particles.radius)
    angle_range = 2π / particles.sidenum
    binsize = angle_range ./ bins
    partition_function = CuArray(zeros(Int32, bins))
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = particlecount(particles) ÷ groups_per_block + 1
    @cuda(threads=numthreads, blocks=numblocks, pmft_angle_pair_parallel!(
        particles, cell_list, partition_function, Float32.(binsize),
        angle_range, rmax, maxcount, groupcount, groups_per_block))
    return Matrix(log.(partition_function))
end

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
        orderfunc::Function; iscomplex::Bool = true, numtype::DataType = Float32,
        returngpu::Bool = false)
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = particlecount(particles) ÷ groups_per_block + 1
    orders = CuArray(iscomplex ? zeros(Complex{numtype}, particlecount(particles))
                               : zeros(numtype, particlecount(particles)))
    @cuda(threads=numthreads, blocks=numblocks,
          shmem = 2 * groups_per_block * (groupcount + neighbors) * sizeof(numtype),
          local_order_parallel!(particles, cell_list, orders, orderfunc, neighbors,
                                maxcount, groupcount, groups_per_block, numtype))
    return returngpu ? orders : Vector(orders)
end

function solidliquid(
        particles::RegularPolygons, cell_list::SeqCellList, katic_orders::Vector;
        threshold::Real = 0.7, rmax::Real = 2 * particles.radius)
    solidbonds = zeros(UInt32, particlecount(particles))
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for idx in cell_list.cells[i, j]
            for neighbor_cell in get_neighbors(cell_list.cells, i, j)
                for neighbor in neighbor_cell
                    dist = √((particles.centers[1, idx] - particles.centers[1, neighbor])^2
                           + (particles.centers[2, idx] - particles.centers[2, neighbor])^2)
                    if dist <= rmax && real(
                            katic_orders[idx] * conj(katic_orders[neighbor])) > threshold
                        solidbonds[idx] += 1
                    end
                end
            end
        end
    end
    return solidbonds
end

function solidliquid(
        particles::RegularPolygons, cell_list::CuCellList, katic_orders::CuVector;
        threshold::Real = 0.7, rmax::Real = 2 * particles.radius)
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = particlecount(particles) ÷ groups_per_block + 1
    solidbonds = CuArray(zeros(Int32, particlecount(particles)))
    @cuda(threads=numthreads, blocks=numblocks, shmem = groups_per_block * sizeof(Int32),
          solidliquid_parallel!(particles, cell_list, katic_orders, solidbonds,
                                threshold, rmax, maxcount, groupcount, groups_per_block))
    return Vector(solidbonds)
end

function local_order_parallel!(particles::RegularPolygons, cell_list::CuCellList,
        orders::CuDeviceVector, orderfunc::Function, neighbors::Integer, maxcount::Integer,
        groupcount::Integer, groups_per_block::Integer, numtype::DataType)
    active_threads = groups_per_block * groupcount
    shared_memory = CuDynamicSharedArray(
        numtype, 2 * groups_per_block * (groupcount + neighbors))
    group_r = @view shared_memory[1:active_threads]
    group_angle = @view shared_memory[active_threads + 1 : 2 * active_threads]
    if neighbors > 0
        neighbor_r = @view shared_memory[
            2 * active_threads + 1 : 2 * active_threads + groups_per_block * neighbors]
        neighbor_angle = @view shared_memory[
            2 * active_threads + groups_per_block * neighbors + 1 : end]
    end

    is_thread_active = threadIdx().x <= active_threads
    group, thread = divrem(threadIdx().x - 1, groupcount)
    group += 1
    if is_thread_active
        particle = (blockIdx().x - 1) * groups_per_block + group
        is_thread_active = particle <= particlecount(particles)
        if is_thread_active
            i, j = get_cell_list_indices(particles, cell_list, particle)
            if neighbors > 0
                is_thread_active = count_neighbors(cell_list.counts, i, j) >= neighbors
                if is_thread_active
                    calc_neighbor!(particles, cell_list, group_r, group_angle,
                                   particle, i, j, maxcount, thread)
                end
            else
                is_thread_active = count_neighbors(cell_list.counts, i, j) > 0
                if is_thread_active
                    calc_order!(particles, cell_list, group_r, group_angle, orderfunc,
                                particle, i, j, maxcount, thread)
                end
            end
        end
    end

    if neighbors > 0
        CUDA.sync_threads()
        k_partition_select!(group_r, group_angle, neighbor_r, neighbor_angle, neighbors,
                            group, thread, groupcount, is_thread_active)
        if is_thread_active && thread == 0
            for iteridx in (group - 1) * neighbors + 1 : group * neighbors
                orders[particle] += orderfunc(neighbor_r[iteridx], neighbor_angle[iteridx])
            end
        end
    else
        if !is_thread_active
            group_r[threadIdx().x] = 0.0f0
            group_angle[threadIdx().x] = 0.0f0
        end
        CUDA.sync_threads()
        sum_parallel_double!(group_r, group_angle)
        if threadIdx().x == 1
            orders[particle] = group_r[1] + 1im * group_angle[1]
        end
    end
    return
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

function calc_order!(particles::RegularPolygons, cell_list::CuCellList,
        group_r::SubArray, group_angle::SubArray, orderfunc::Function,
        particle::Integer, i::Integer, j::Integer, maxcount::Integer, thread::Integer)
    ineighbor, jneighbor, kneighbor = get_neighbor_indices(
        cell_list, thread, maxcount, i, j)
    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        if particle != neighbor
            r, theta = get_dist_and_angle(particles, particle, neighbor)
            order = orderfunc(r, theta)
            group_r[threadIdx().x] = real(order)
            group_angle[threadIdx().x] = imag(order)
        else
            group_r[threadIdx().x] = 0.0f0
            group_angle[threadIdx().x] = 0.0f0
        end
    else
        group_r[threadIdx().x] = 0.0f0
        group_angle[threadIdx().x] = 0.0f0
    end
    return
end

function solidliquid_parallel!(particles::RegularPolygons, cell_list::CuCellList,
        katic_orders::CuDeviceVector, solidbonds::CuDeviceVector, threshold::Real,
        rmax::Real, maxcount::Integer, groupcount::Integer, groups_per_block::Integer)
    blockbonds = CuDynamicSharedArray(Int32, groups_per_block)

    is_thread_active = threadIdx().x <= groups_per_block * groupcount
    group, thread = divrem(threadIdx().x - 1, groupcount)
    group += 1
    if is_thread_active
        idx = (blockIdx().x - 1) * groups_per_block + group
        is_thread_active = idx <= particlecount(particles)
        if is_thread_active && thread == 0
            blockbonds[group] = 0
        end
    end

    CUDA.sync_threads()
    if is_thread_active
        i, j = get_cell_list_indices(particles, cell_list, idx)
        ineighbor, jneighbor, kneighbor = get_neighbor_indices(
            cell_list, thread, maxcount, i, j)
        if kneighbor <= cell_list.counts[ineighbor, jneighbor]
            neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
            if idx != neighbor
                dist = √((particles.centers[1, idx] - particles.centers[1, neighbor])^2
                    + (particles.centers[2, idx] - particles.centers[2, neighbor])^2)
                if (dist <= rmax && real(katic_orders[idx]
                        * conj(katic_orders[neighbor])) > threshold)
                    CUDA.@atomic blockbonds[group] += 1
                end
            end
        end
    end

    CUDA.sync_threads()
    if is_thread_active && thread == 0
        solidbonds[idx] = blockbonds[group]
    end
    return
end

function pmft_angle_pair_parallel!(particles::RegularPolygons, cell_list::CuCellList,
        partition_function::CuDeviceMatrix, binsize::Tuple{<:Real, <:Real},
        angle_range::Real, rmax::Real, maxcount::Integer, groupcount::Integer,
        groups_per_block::Integer)
    if threadIdx().x <= groups_per_block * groupcount
        group, thread = divrem(threadIdx().x - 1, groupcount)
        group += 1
        idx = (blockIdx().x - 1) * groups_per_block + group
        if idx <= particlecount(particles)
            i, j = get_cell_list_indices(particles, cell_list, idx)
            ineighbor, jneighbor, kneighbor = get_neighbor_indices(
                cell_list, thread, maxcount, i, j)
            if kneighbor <= cell_list.counts[ineighbor, jneighbor]
                neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
                if idx != neighbor
                    r, theta = get_dist_and_angle(particles, idx, neighbor)
                    if r <= rmax
                        theta1 = mod(particles.angles[idx] - theta, angle_range)
                        theta2 = mod(particles.angles[neighbor] - theta + π, angle_range)
                        bin1 = Int(theta1 ÷ binsize[1]) + 1
                        bin2 = Int(theta2 ÷ binsize[2]) + 1
                        CUDA.@atomic partition_function[bin1, bin2] += 1
                        CUDA.@atomic partition_function[bin2, bin1] += 1
                    end
                end
            end
        end
    end
    return
end

function get_dist_and_angle(particles::RegularPolygons, i::Integer, j::Integer)
    rij = apply_parallelogram_boundary(particles,
        (particles.centers[1, j] - particles.centers[1, i],
         particles.centers[2, j] - particles.centers[2, i]))
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
                    group_r[thread], group_r[thread + step] = (
                        group_r[thread + step], group_r[thread])
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

@inline function sum_parallel_double!(a::SubArray, b::SubArray)
    step = length(a) ÷ 2
    while step != 0
        if threadIdx().x <= step
            a[threadIdx().x] += a[threadIdx().x + step]
            b[threadIdx().x] += b[threadIdx().x + step]
        end
        CUDA.sync_threads()
        step ÷= 2
    end
end

@inline function count_neighbors(counts::AbstractMatrix, i::Integer, j::Integer)
    (counts[mod(i - 2, size(counts, 1)) + 1, mod(j - 2, size(counts, 2)) + 1]
     + counts[mod(i - 2, size(counts, 1)) + 1, j]
     + counts[mod(i - 2, size(counts, 1)) + 1, mod(j, size(counts, 2)) + 1]
     + counts[i, mod(j - 2, size(counts, 2)) + 1]
     + counts[i, j] - 1
     + counts[i, mod(j, size(counts, 2)) + 1]
     + counts[mod(i, size(counts, 1)) + 1, mod(j - 2, size(counts, 2)) + 1]
     + counts[mod(i, size(counts, 1)) + 1, j]
     + counts[mod(i, size(counts, 1)) + 1, mod(j, size(counts, 2)) + 1])
end

@inline function get_neighbors(cells::AbstractMatrix, i::Integer, j::Integer)
    (cells[mod(i - 2, size(cells, 1)) + 1, mod(j - 2, size(cells, 2)) + 1],
     cells[mod(i - 2, size(cells, 1)) + 1, j],
     cells[mod(i - 2, size(cells, 1)) + 1, mod(j, size(cells, 2)) + 1],
     cells[i, mod(j - 2, size(cells, 2)) + 1],
     cells[i, j],
     cells[i, mod(j, size(cells, 2)) + 1],
     cells[mod(i, size(cell_list.counts, 1)) + 1, mod(j - 2, size(cells, 2)) + 1],
     cells[mod(i, size(cells, 1)) + 1, j],
     cells[mod(i, size(cells, 1)) + 1, mod(j, size(cells, 2)) + 1])
end
