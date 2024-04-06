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
