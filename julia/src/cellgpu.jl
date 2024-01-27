struct CuCellList{T<:Real, A<:AbstractArray, M<:AbstractMatrix,
                  V<:AbstractVector} <: CellList
    width::Tuple{T, T}
    shift::V

    cells::A
    counts::M
end

function CuCellList(particles::RegularPolygons, shift::AbstractArray = [0.0f0, 0.0f0];
                    maxwidth::Real = 0.0f0, max_particle_per_cell=20)
    if iszero(maxwidth)
        maxwidth = get_maxwidth(particles)
    end
    boxsize = Array(particles.boxsize)
    width = (boxsize[1] / ceil(boxsize[1] / maxwidth),
             boxsize[2] / ceil(boxsize[2] / maxwidth))
    m, n = Int(boxsize[1] ÷ width[1]), Int(boxsize[2] ÷ width[2])
    cells = Array{Int32, 3}(undef, max_particle_per_cell, m, n)
    counts = zeros(Int32, m, n)

    cells = CuArray(cells)
    counts = CuArray(counts)

    CUDA.@allowscalar @cuda(
        threads=numthreads, blocks=particlecount(particles)÷numthreads+1,
        build_cells_parallel!(particles, cells, counts, width, shift[1], shift[2]))

    shift = CuVector{eltype(particles.centers)}(shift)
    CuCellList{eltype(width), typeof(cells), typeof(counts), typeof(shift)}(
        width, shift, cells, counts)
end

Adapt.@adapt_structure CuCellList

@inline get_maxwidth(particles::RegularPolygons) = 
    particles.sidenum <= 4 ? 2 * particles.radius : 2 * √2 * particles.radius

@inline function get_cell_list_indices(particles::RegularPolygons,
        gridsize::Tuple{<:Integer, <:Integer}, width::Tuple{<:Real, <:Real},
        xshift::Real, yshift::Real, idx::Integer)
    shearshift = particles.boxsize[1] / 2 - particles.centers[2, idx] * particles.boxshear[]
    (mod(floor(Int, (particles.centers[1, idx] + shearshift - xshift)
        / width[1]), gridsize[1]) + 1,
     mod(floor(Int, (particles.centers[2, idx] + particles.boxsize[2] / 2 - yshift)
        / width[2]), gridsize[2]) + 1)
end

@inline function get_cell_list_indices(
        particles::RegularPolygons, cell_list::CuCellList, idx::Integer)
    get_cell_list_indices(particles, size(cell_list.counts), cell_list.width,
                          cell_list.shift[1], cell_list.shift[2], idx)
end

function build_cells_parallel!(particles::RegularPolygons, cells::CuDeviceArray,
        counts::CuDeviceArray, width::NTuple{2, <:Real}, xshift::Real, yshift::Real)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= particlecount(particles)
        i, j = get_cell_list_indices(particles, size(counts), width, xshift, yshift, idx)
        cellidx = CUDA.@atomic counts[i, j] += 1
        cells[cellidx + 1, i, j] = idx
    end
    return
end

function shift_cells!(particles::RegularPolygons, cell_list::CuCellList,
                      direction::Tuple{<:Integer, <:Integer}, shift::Real)
    CUDA.allowscalar() do
        cell_list.shift[1] += direction[1] * shift
        cell_list.shift[2] += direction[2] * shift
        cell_list.counts .= 0
        @cuda(threads=numthreads, blocks=particlecount(particles)÷numthreads+1,
            build_cells_parallel!(particles, cell_list.cells, cell_list.counts,
                cell_list.width, cell_list.shift[1], cell_list.shift[2]))
    end
end

function count_overlaps(particles::RegularPolygons, cell_list::CuCellList)
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = length(cell_list.counts) ÷ groups_per_block + 1
    overlapcount = CUDA.zeros(Int32)
    @cuda(threads=numthreads, blocks=numblocks,
          shmem = groups_per_block * groupcount * sizeof(Int32),
          count_overlaps_parallel!(particles, cell_list, overlapcount, maxcount,
                                   groupcount, groups_per_block))
    return CUDA.@allowscalar overlapcount[]
end

function has_overlap(particles::RegularPolygons, cell_list::CuCellList)
    return count_overlaps(particles, cell_list) > 0
end

function calculate_potentials!(particles::RegularPolygons, cell_list::CuCellList,
        potentials::CuArray, potential::Union{Function, Nothing},
        pairpotential::Union{Function, Nothing})
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = numthreads ÷ groupcount
    numblocks = length(cell_list.counts) ÷ groups_per_block + 1
    @cuda(threads=numthreads, blocks=numblocks,
          calculate_potentials_parallel!(particles, cell_list, potentials,
          potential, pairpotential, maxcount, groupcount, groups_per_block))
end

function count_overlaps_parallel!(particles::RegularPolygons, cell_list::CuCellList,
        overlapcount::CuDeviceArray, maxcount::Integer, groupcount::Integer,
        groups_per_block::Integer)
    block_overlaps = CuDynamicSharedArray(Int32, groupcount * groups_per_block)

    if threadIdx().x <= length(block_overlaps)
        group, thread = divrem(threadIdx().x - 1, groupcount)
        group += 1
        cell = (blockIdx().x - 1) * groups_per_block + group
        j, i = divrem(cell - 1, size(cell_list.counts, 1))
        i += 1
        j += 1
        if cell <= length(cell_list.counts) && cell_list.counts[i, j] != 0
            block_overlaps[threadIdx().x] = count_neighbor_overlaps!(
                particles, cell_list, i, j, maxcount, thread)
        else
            block_overlaps[threadIdx().x] = 0
        end
    end
    CUDA.sync_threads()

    sum_parallel!(block_overlaps)
    if threadIdx().x == 1
        CUDA.@atomic overlapcount[] += block_overlaps[1]
    end
    return
end

function count_neighbor_overlaps!(particles::RegularPolygons, cell_list::CuCellList,
        i::Integer, j::Integer, maxcount::Integer, thread::Integer)
    thread_overlaps = zero(Int32)
    ineighbor, jneighbor, kneighbor = get_neighbor_indices(
        cell_list, thread, maxcount, i, j)
    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        for k in 1:cell_list.counts[i, j]
            idx = cell_list.cells[k, i, j]
            if idx != neighbor
                if is_overlapping(particles, idx, neighbor)
                    thread_overlaps += 1
                end
            end
        end
    end
    return thread_overlaps
end

function sum_parallel!(a::CuDeviceArray)
    step = length(a) ÷ 2
    while step != 0
        if threadIdx().x <= step
            a[threadIdx().x] += a[threadIdx().x + step]
        end
        CUDA.sync_threads()
        step ÷= 2
    end
end

function calculate_potentials_parallel!(particles::RegularPolygons, cell_list::CuCellList,
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
            calculate_neighbor_potentials!(particles, cell_list, potentials, potential,
                                           pairpotential, i, j, maxcount, thread)
        else
            is_thread_active = false
        end
    end
    return
end

function calculate_neighbor_potentials!(particles::RegularPolygons, cell_list::CuCellList,
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
                    CUDA.@atomic potentials[idx] += pairpotential(particles, idx, neighbor)
                end
            elseif !isnothing(potential)
                CUDA.@atomic potentials[idx] += potential(particles, idx)
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
