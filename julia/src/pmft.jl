function pmft_angle_pair(particles::RegularPolygons;
        bins::Tuple{<:Integer, <:Integer} = (1024, 1024),
        rmax::Real = 2.5 * particles.radius, neighbors::Integer = 0)
    if isa(particles.centers, CuArray)
        return pmft_angle_pair(particles, CuCellList(particles),
            bins=bins, rmax=rmax, neighbors=neighbors)
    else
        return pmft_angle_pair(particles, SeqCellList(particles),
            bins=bins, rmax=rmax, neighbors=neighbors)
    end
end

function pmft_angle_pair(particles::RegularPolygons, cell_list::SeqCellList;
        bins::Tuple{<:Integer, <:Integer} = (1024, 1024),
        rmax::Real = 2.5 * particles.radius, neighbors::Integer = 0)
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
                    end
                end
            end
        end
    end

    return Matrix(log.(partition_function))
end

function pmft_angle_pair(particles::RegularPolygons, cell_list::CuCellList;
        bins::Tuple{<:Integer, <:Integer} = (1024, 1024),
        rmax::Real = 2.5 * particles.radius, neighbors::Integer = 0)
    angle_range = 2π / particles.sidenum
    binsize = angle_range ./ bins
    partition_function = CuArray(zeros(Int32, bins))
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = particlecount(particles) ÷ groups_per_block + 1
    numtype = eltype(particles.angles)
    if neighbors > 0
        @cuda(threads=numthreads, blocks=numblocks,
            shmem = groups_per_block * (3 * groupcount + 2 * neighbors) * sizeof(numtype),
            pmft_angle_pair_nn_parallel!(
                particles, cell_list, partition_function, binsize, angle_range,
                neighbors, maxcount, groupcount, groups_per_block))
    else
        @cuda(threads=numthreads, blocks=numblocks, pmft_angle_pair_parallel!(
            particles, cell_list, partition_function, binsize,
            angle_range, rmax, maxcount, groupcount, groups_per_block))
    end
    return Matrix(log.(partition_function))
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
                        pmft_angle_pair_single!(partition_function, binsize, theta,
                            particles.angles[idx], particles.angles[neighbor], angle_range)
                    end
                end
            end
        end
    end
    return
end

function pmft_angle_pair_nn_parallel!(particles::RegularPolygons, cell_list::CuCellList,
        partition_function::CuDeviceMatrix, binsize::Tuple{<:Real, <:Real},
        angle_range::Real, neighbors::Integer, maxcount::Integer,
        groupcount::Integer, groups_per_block::Integer)
    numtype = eltype(particles.angles)
    active_threads = groups_per_block * groupcount
    shared_memory = CuDynamicSharedArray(
        numtype, groups_per_block * (3 * groupcount + 2 * neighbors))
    group_r = @view shared_memory[1:active_threads]
    group_orientation = @view shared_memory[active_threads + 1 : 2 * active_threads]
    group_angle = @view shared_memory[2 * active_threads + 1 : 3 * active_threads]
    neighbor_orientation = @view shared_memory[
        3 * active_threads + 1 : 3 * active_threads + groups_per_block * neighbors]
    neighbor_angle = @view shared_memory[
        3 * active_threads + groups_per_block * neighbors + 1 : end]

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
                calc_full_neighbor!(particles, cell_list, group_r, group_orientation,
                                    group_angle, particle, i, j, maxcount, thread)
            else
                calc_partial_neighbor!(particles, cell_list, partition_function, binsize,
                                       angle_range, particle, i, j, maxcount, thread)
            end
        end
    end

    CUDA.sync_threads()
    k_partition_select_double!(
        group_r, group_orientation, group_angle, neighbor_orientation,
        neighbor_angle, neighbors, group, thread, groupcount, is_thread_active)
    if is_thread_active && thread < neighbors
        pmft_angle_pair_single!(partition_function, binsize, neighbor_angle[thread + 1],
            particles.angles[particle], neighbor_orientation[thread + 1], angle_range)
    end
    return
end

function pmft_angle_pair_single!(
        partition_function::CuDeviceMatrix, binsize::Tuple{<:Real, <:Real},
        theta::Real, angle1::Real, angle2::Real, angle_range::Real)
    theta1 = mod(angle1 - theta, angle_range)
    theta2 = mod(angle2 - theta + π, angle_range)
    bin1 = Int(theta1 ÷ binsize[1]) + 1
    bin2 = Int(theta2 ÷ binsize[2]) + 1
    CUDA.@atomic partition_function[bin1, bin2] += 1
    return
end

function calc_full_neighbor!(particles::RegularPolygons, cell_list::CuCellList,
        group_r::SubArray, group_orientation::SubArray, group_angle::SubArray,
        particle::Integer, i::Integer, j::Integer, maxcount::Integer, thread::Integer)
    ineighbor, jneighbor, kneighbor = get_neighbor_indices(
        cell_list, thread, maxcount, i, j)
    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        if particle != neighbor
            group_r[threadIdx().x], group_angle[threadIdx().x] = get_dist_and_angle(
                particles, particle, neighbor)
            group_orientation[threadIdx().x] = particles.angles[neighbor]
        else
            group_r[threadIdx().x] = typemax(eltype(group_r))
        end
    else
        group_r[threadIdx().x] = typemax(eltype(group_r))
    end
    return
end

function calc_partial_neighbor!(particles::RegularPolygons, cell_list::CuCellList,
        partition_function::CuDeviceMatrix, binsize::Tuple{<:Real, <:Real},
        angle_range::Real, particle::Integer, i::Integer, j::Integer,
        maxcount::Integer, thread::Integer)
    ineighbor, jneighbor, kneighbor = get_neighbor_indices(
        cell_list, thread, maxcount, i, j)
    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        if particle != neighbor
            _, theta = get_dist_and_angle(particles, particle, neighbor)
            pmft_angle_pair_single!(partition_function, binsize, theta,
                particles.angles[particle], particles.angles[neighbor], angle_range)
        end
    end
    return
end

function k_partition_select_double!(group_r::SubArray, group_angle1::SubArray,
        group_angle2::SubArray, neighbor_angle1::SubArray, neighbor_angle2::SubArray,
        k::Integer, group::Integer, thread::Integer, groupcount::Integer,
        is_thread_active::Bool)
    if is_thread_active
        group_r = @view group_r[(group - 1) * groupcount + 1 : group * groupcount]
        group_angle1 = @view group_angle1[(group - 1) * groupcount + 1 : group * groupcount]
        group_angle2 = @view group_angle2[(group - 1) * groupcount + 1 : group * groupcount]
        neighbor_angle1 = @view neighbor_angle1[(group - 1) * k + 1 : group * k]
        neighbor_angle2 = @view neighbor_angle2[(group - 1) * k + 1 : group * k]
        thread += 1
    end
    for selection in 1:k
        step = 1
        while step < groupcount
            if is_thread_active && (thread - 1) % 2step == 0 && thread + step <= groupcount
                if group_r[thread] > group_r[thread + step]
                    group_r[thread], group_r[thread + step] = (
                        group_r[thread + step], group_r[thread])
                    group_angle1[thread], group_angle1[thread + step] = (
                        group_angle1[thread + step], group_angle1[thread])
                    group_angle2[thread], group_angle2[thread + step] = (
                        group_angle2[thread + step], group_angle2[thread])
                end
            end
            step *= 2
            CUDA.sync_threads()
        end
        if is_thread_active && isone(thread)
            neighbor_angle1[selection] = group_angle1[1]
            neighbor_angle2[selection] = group_angle2[1]
            group_r[1] = typemax(eltype(group_r))
        end
    end
    return
end
