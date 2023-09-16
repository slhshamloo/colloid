struct CuCellList{T<:Real, A<:AbstractArray, M<:AbstractMatrix,
                  V<:AbstractVector} <: CellList
    width::Tuple{T, T}
    shift::V

    cells::A
    counts::M
end

function CuCellList(colloid::Colloid, shift::AbstractArray = [0.0f0, 0.0f0];
                    maxwidth::Real = 0.0f0, max_particle_per_cell=20)
    if iszero(maxwidth)
        maxwidth = colloid.sidenum <= 4 ? 2 * colloid.radius : 2 * √2 * colloid.radius
    end
    boxsize = Array(colloid.boxsize)
    width = (boxsize[1] / ceil(boxsize[1] / maxwidth),
             boxsize[2] / ceil(boxsize[2] / maxwidth))
    m, n = Int(boxsize[1] ÷ width[1]), Int(boxsize[2] ÷ width[2])
    cells = Array{Int32, 3}(undef, max_particle_per_cell, m, n)
    counts = zeros(Int32, m, n)

    cells = CuArray(cells)
    counts = CuArray(counts)
    @cuda(threads=numthreads, blocks=particle_count(colloid)÷numthreads+1,
          build_cells_parallel!(colloid, cells, counts, width, shift[1], shift[2]))

    shift = CuVector{eltype(colloid.centers)}(shift)
    CuCellList{eltype(width), typeof(cells), typeof(counts), typeof(shift)}(
        width, shift, cells, counts)
end

Adapt.@adapt_structure CuCellList

@inline function get_cell_list_indices(colloid::Colloid,
        gridsize::Tuple{<:Integer, <:Integer}, width::Tuple{<:Real, <:Real},
        xshift::Real, yshift::Real, idx::Integer)
    (mod(floor(Int, (colloid.centers[1, idx] + colloid.boxsize[1] / 2 - xshift)
        / width[1]), gridsize[1]) + 1,
     mod(floor(Int, (colloid.centers[2, idx] + colloid.boxsize[2] / 2 - yshift)
        / width[2]), gridsize[2]) + 1)
end

@inline get_cell_list_indices(colloid::Colloid, cell_list::CuCellList, idx::Integer) =
    get_cell_list_indices(colloid, size(cell_list.counts), cell_list.width,
                          cell_list.shift[1], cell_list.shift[2], idx)

function build_cells_parallel!(colloid::Colloid, cells::CuDeviceArray,
        counts::CuDeviceArray, width::NTuple{2, <:Real}, xshift::Real, yshift::Real)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= particle_count(colloid)
        i, j = get_cell_list_indices(colloid, size(counts), width, xshift, yshift, idx)
        cellidx = CUDA.@atomic counts[i, j] += 1
        cells[cellidx + 1, i, j] = idx
    end
    return
end

function shift_cells!(colloid::Colloid, cell_list::CuCellList,
                      direction::Tuple{<:Integer, <:Integer}, shift::Real)
    cell_list.shift[1] += direction[1] * shift
    cell_list.shift[2] += direction[2] * shift
    cell_list.counts .= 0
    @cuda(threads=numthreads, blocks=particle_count(colloid)÷numthreads+1,
          build_cells_parallel!(colloid, cell_list.cells, cell_list.counts,
                                cell_list.width, cell_list.shift[1], cell_list.shift[2]))
end

function count_overlaps(colloid::Colloid, cell_list::CuCellList)
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = length(cell_list.counts) ÷ groups_per_block + 1
    overlapcounts = zero(cell_list.counts)
    @cuda(threads=numthreads, blocks=numblocks,
          shmem = groups_per_block * sizeof(Int32),
          count_overlaps_parallel!(colloid, cell_list, overlapcounts, maxcount,
                                   groupcount, groups_per_block))
    return sum(overlapcounts) ÷ 2
end

function has_overlap(colloid::Colloid, cell_list::CuCellList)
    return count_overlaps(colloid, cell_list) > 0
end

function calculate_potentials!(colloid::Colloid, cell_list::CuCellList,
        potentials::CuArray, potential::Union{Function, Nothing},
        pairpotential::Union{Function, Nothing})
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = length(cell_list.counts) ÷ groups_per_block + 1
    @cuda(threads=numthreads, blocks=numblocks,
          calculate_potentials_parallel!(colloid, cell_list, potentials,
          potential, pairpotential, maxcount, groupcount, groups_per_block))
end

function count_overlaps_parallel!(colloid::Colloid, cell_list::CuCellList,
        overlapcounts::CuDeviceMatrix, maxcount::Integer, groupcount::Integer,
        groups_per_block::Integer)
    group_overlaps = CuDynamicSharedArray(Int32, groups_per_block)
    is_thread_active = threadIdx().x <= groups_per_block * groupcount

    if is_thread_active
        group, thread = divrem(threadIdx().x - 1, groupcount)
        group += 1
        if thread == 0
            group_overlaps[group] = 0
        end
    end
    CUDA.sync_threads()

    if is_thread_active
        cell = (blockIdx().x - 1) * groups_per_block + group
        j, i = divrem(cell - 1, size(cell_list.counts, 1))
        i += 1
        j += 1
        if cell <= length(cell_list.counts) && cell_list.counts[i, j] != 0
            count_neighbor_overlaps!(colloid, cell_list, group_overlaps,
                                     i, j, maxcount, group, thread)
        else
            is_thread_active = false
        end
    end
    CUDA.sync_threads()

    if is_thread_active && thread == 0
        overlapcounts[i, j] = group_overlaps[group]
    end
    return
end

function count_neighbor_overlaps!(colloid::Colloid, cell_list::CuCellList,
        group_overlaps::CuDeviceVector, i::Integer, j::Integer, maxcount::Integer,
        group::Integer, thread::Integer)
    ineighbor, jneighbor, kneighbor = get_neighbor_indices(
        cell_list, thread, maxcount, i, j)
    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        overlap_count = 0
        for k in 1:cell_list.counts[i, j]
            idx = cell_list.cells[k, i, j]
            if idx != neighbor
                if is_overlapping(colloid, idx, neighbor)
                    overlap_count += 1
                end
            end
        end
        CUDA.@atomic group_overlaps[group] += overlap_count
    end
end

function calculate_potentials_parallel!(colloid::Colloid, cell_list::CuCellList,
        potentials::CuDeviceArray, potential::Union{Function, Nothing},
        pairpotential::Union{Function, Nothing}, maxcount::Integer,
        groupcount::Integer, groups_per_block::Integer)
    is_thread_active = threadIdx().x <= groups_per_block * groupcount
    if is_thread_active
        group, thread = divrem(threadIdx().x - 1, groupcount)
        group += 1
    end
    CUDA.sync_threads()

    if is_thread_active
        cell = (blockIdx().x - 1) * groups_per_block + group
        j, i = divrem(cell - 1, size(cell_list.counts, 1))
        i += 1
        j += 1
        if cell <= length(cell_list.counts) && cell_list.counts[i, j] != 0
            calculate_neighbor_potentials!(colloid, cell_list, potentials, potential,
                                           pairpotential, i, j, maxcount, thread)
        else
            is_thread_active = false
        end
    end
    return
end

function calculate_neighbor_potentials!(colloid::Colloid, cell_list::CuCellList,
        potentials::CuDeviceArray, potential::Union{Function, Nothing},
        pairpotential::Union{Function, Nothing}, i::Integer, j::Integer,
        maxcount::Integer, thread::Integer)
    ineighbor, jneighbor, kneighbor = get_neighbor_indices(
        cell_list, thread, maxcount, i, j)
    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        for k in 1:cell_list.counts[i, j]
            idx = cell_list.cells[k, i, j]
            if idx != neighbor
                if !isnothing(pairpotential)
                    CUDA.@atomic potentials[idx] += pairpotential(colloid, idx, neighbor)
                end
            elseif !isnothing(potential)
                CUDA.@atomic potentials[idx] += potential(colloid, idx)
            end
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
