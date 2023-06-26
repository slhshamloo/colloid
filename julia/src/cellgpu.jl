struct CuCellList{T<:Real, A<:AbstractArray, M<:AbstractMatrix,
                  V<:AbstractVector} <: CellList
    width::Tuple{T, T}
    shift::V

    cells::A
    counts::M

    _temp_cells::A
    _temp_counts::M
end

function CuCellList(colloid::Colloid, shift::AbstractArray = [0.0f0, 0.0f0];
                    maxwidth::Real = 0.0f0, max_particle_per_cell=20)
    if iszero(maxwidth)
        maxwidth = colloid.sidenum <= 4 ? 2 * colloid.radius : 2 * √2 * colloid.radius
    end
    width = (colloid.boxsize[1] / ceil(colloid.boxsize[1] / maxwidth),
             colloid.boxsize[2] / ceil(colloid.boxsize[2] / maxwidth))
    m, n = Int(colloid.boxsize[1] ÷ width[1]), Int(colloid.boxsize[2] ÷ width[2])
    cells = Array{Int32, 3}(undef, max_particle_per_cell, m, n)
    counts = zeros(Int32, m, n)

    for idx in 1:particle_count(colloid)
        i, j = get_cell_list_indices(colloid, (m, n), width, shift, idx)
        counts[i, j] += 1
        cells[counts[i, j], i, j] = idx
    end

    shift = CuVector{eltype(colloid.centers)}(shift)
    cells = CuArray(cells)
    counts = CuArray(counts)
    CuCellList{eltype(width), typeof(cells), typeof(counts), typeof(shift)}(
        width, shift, cells, counts, similar(cells), zero(counts))
end

Adapt.@adapt_structure CuCellList

@inline function get_cell_list_indices(colloid::Colloid,
        gridsize::Tuple{<:Integer, <:Integer}, width::Tuple{<:Real, <:Real},
        shift::AbstractArray, idx::Integer)
    (mod(floor(Int, (colloid.centers[1, idx] + colloid.boxsize[1] / 2 - shift[1])
        / width[1]), gridsize[1]) + 1,
     mod(floor(Int, (colloid.centers[2, idx] + colloid.boxsize[2] / 2 - shift[2])
        / width[2]), gridsize[2]) + 1)
end

@inline function get_cell_list_indices(colloid::Colloid, cell_list::CuCellList,
                                       idx::Integer)
    get_cell_list_indices(colloid, size(cell_list.counts), cell_list.width,
                          cell_list.shift, idx)
end

function shift_cells!(colloid::Colloid, cell_list::CuCellList,
                      direction::Tuple{<:Integer, <:Integer}, shift::Real)
    cell_list.shift[1] += direction[1] * shift
    cell_list.shift[2] += direction[2] * shift
    numblocks = (size(cell_list.cells, 2) ÷ numthreads[1] + 1,
                 size(cell_list.cells, 3) ÷ numthreads[2] + 1)
    @cuda threads=numthreads blocks=numblocks shift_cell!(colloid, cell_list, direction)
    cell_list.cells .= cell_list._temp_cells
    cell_list.counts .= cell_list._temp_counts
    cell_list._temp_counts .= 0
end

function shift_cell!(colloid::Colloid, cell_list::CuCellList,
                     direction::Tuple{<:Integer, <:Integer})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i > size(cell_list.cells, 2) || j > size(cell_list.cells, 3)
        return
    end

    shiftij!(colloid, cell_list, i, j, i, j)
    i_neighbor = mod(i + direction[1] - 1, size(cell_list.cells, 2)) + 1
    j_neighbor = mod(j + direction[2] - 1, size(cell_list.cells, 3)) + 1
    shiftij!(colloid, cell_list, i, j, i_neighbor, j_neighbor)
    return
end

@inline function shiftij!(colloid::Colloid, cell_list::CuCellList,
                          i::Integer, j::Integer, ii::Integer, jj::Integer)
    for k in 1:cell_list.counts[ii, jj]
        idx = cell_list.cells[k, ii, jj]
        if (i, j) == get_cell_list_indices(colloid, cell_list, idx)
            cell_list._temp_counts[i, j] += 1
            cell_list._temp_cells[cell_list._temp_counts[i, j], i, j] = idx
        end
    end
end

function count_overlaps(colloid::Colloid, cell_list::CuCellList)
    blockthreads = (numthreads[1] * numthreads[2])
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = blockthreads ÷ groupcount
    numblocks = length(cell_list.counts) ÷ groups_per_block + 1
    overlapcounts = zero(cell_list.counts)
    @cuda(threads=blockthreads, blocks=numblocks,
          shmem = groups_per_block * sizeof(Int32),
          count_overlaps_parallel(colloid, cell_list, overlapcounts, maxcount,
                                  groupcount, groups_per_block))
    return sum(overlapcounts) ÷ 2
end

function has_overlap(colloid::Colloid, cell_list::CuCellList)
    return count_overlaps(colloid, cell_list) > 0
end

function count_overlaps_parallel(colloid::Colloid, cell_list::CuCellList,
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
            count_neighbor_overlaps(colloid, cell_list, group_overlaps,
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

function count_neighbor_overlaps(colloid::Colloid, cell_list::CuCellList,
        group_overlaps::CuDeviceVector, maxcount::Integer, i::Integer, j::Integer,
        group::Integer, thread::Integer)
    relpos, kneighbor = divrem(thread, maxcount)
    kneighbor += 1
    jdelta, idelta = divrem(relpos, 3)
    ineighbor = mod(i + idelta - 2, size(cell_list.cells, 2)) + 1
    jneighbor = mod(i + jdelta - 2, size(cell_list.cells, 3)) + 1

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
