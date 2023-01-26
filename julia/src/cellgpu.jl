struct CuCellList{T<:Real, A<:AbstractArray, M<:AbstractMatrix,
                  V<:AbstractVector} <: CellList
    width::Tuple{T, T}
    shift::V

    cells::A
    counts::M

    _temp_cells::A
    _temp_counts::M
end

function CuCellList(colloid::Colloid, shift::AbstractArray = [0.0f0, 0.0f0])
    w = colloid.sidenum <= 4 ? 2 * colloid.radius : 2 * √2 * colloid.radius
    width = (colloid.boxsize[1] / ceil(colloid.boxsize[1] / w),
             colloid.boxsize[2] / ceil(colloid.boxsize[2] / w))

    m, n = Int(colloid.boxsize[1] ÷ width[1]), Int(colloid.boxsize[2] ÷ width[2])
    cells = Array{Int32, 3}(undef, 5, m, n)
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
    (Int((colloid.centers[1, idx] + colloid.boxsize[1] / 2 - shift[1])
         ÷ width[1] + gridsize[1]) % gridsize[1] + 1,
    Int((colloid.centers[2, idx] + colloid.boxsize[2] / 2 - shift[2])
        ÷ width[2] + gridsize[2]) % gridsize[2] + 1)
end

@inline function get_cell_list_indices(colloid::Colloid, cell_list::CuCellList,
                                       idx::Integer)
    get_cell_list_indices(colloid, size(cell_list.counts), cell_list.width,
                          cell_list.shift, idx)
end

function shift_cells!(colloid::Colloid, cell_list::CuCellList,
                      direction::Tuple{<:Integer,  <:Integer}, shift::Real)
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
    lx, ly = size(cell_list.counts)
    if i > lx || j > ly return end

    shiftij!(colloid, cell_list, i, j, i, j)
    i_neighbor = (i + direction[1] - 1 + lx) % lx + 1
    j_neighbor = (j + direction[2] - 1 + ly) % ly + 1
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

function has_overlap(colloid::Colloid, cell_list::CuCellList)
    for cell in CartesianIndices(cell_list.counts)
        i, j = Tuple(cell)
        for m in 1:cell_list.counts[i, j]
            idx = cell_list.cells[m, i, j]
            for n in m+1:cell_list.counts[i, j]
                if is_overlapping(colloid, idx, cell_list.cells[n, i, j])
                    return true
                end
            end
            if (i + j) % 2 == 0
                if has_orthogonal_overlap(colloid, cell_list, i, j, idx)
                    return true
                end
            end
            if i % 2 == 0
                if has_diagonal_overlap(colloid, cell_list, i, j, idx)
                    return true
                end
            end
        end
    end
    return false
end

function count_overlaps(colloid::Colloid, cell_list::CuCellList)
    overlap_count = 0
    for idx in CartesianIndices(cell_list.counts)
        i, j = Tuple(idx)
        for m in 1:cell_list.counts[i, j]
            idx = cell_list.cells[m, i, j]
            for n in m+1:cell_list.counts[i, j]
                if is_overlapping(colloid, idx, cell_list.cells[n, i, j])
                    overlap_count += 1
                end
            end
            if (i + j) % 2 == 0
                overlap_count += count_orthogonal_overlaps(colloid, cell_list, i, j, idx)
            end
            if i % 2 == 0
                overlap_count += count_diagonal_overlaps(colloid, cell_list, i, j, idx)
            end
        end
    end
    return overlap_count
end

@inline function has_orthogonal_overlap(colloid::Colloid, cell_list::CuCellList,
                                        i::Integer, j::Integer, idx::Integer)
    lx, ly = size(cell_list.counts)

    ii = (i == 1 ? lx : i - 1)
    for n in 1:cell_list.counts[ii, j]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, j]) return true end
    end

    jj = (j == 1 ? ly : j - 1)
    for n in 1:cell_list.counts[i, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, i, jj]) return true end
    end

    ii = (i == lx ? 1 : i + 1)
    for n in 1:cell_list.counts[ii, j]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, j]) return true end
    end

    jj = (j == ly ? 1 : j + 1)
    for n in 1:cell_list.counts[i, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, i, jj]) return true end
    end

    return false
end

@inline function has_diagonal_overlap(colloid::Colloid, cell_list::CuCellList,
                                      i::Integer, j::Integer, idx::Integer)
    lx, ly = size(cell_list.counts)

    ii, jj = (i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)
    for n in 1:cell_list.counts[ii, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, jj]) return true end
    end

    ii, jj = (i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)
    for n in 1:cell_list.counts[ii, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, jj]) return true end
    end

    ii, jj = (i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)
    for n in 1:cell_list.counts[ii, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, jj]) return true end
    end

    ii, jj = (i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)
    for n in 1:cell_list.counts[ii, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, jj]) return true end
    end

    return false
end

@inline function count_orthogonal_overlaps(colloid::Colloid, cell_list::CuCellList,
                                           i::Integer, j::Integer, idx::Integer)
    count = 0
    lx, ly = size(cell_list.counts)

    ii = (i == 1 ? lx : i - 1)
    for n in 1:cell_list.counts[ii, j]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, j]) count += 1 end
    end

    jj = (j == 1 ? ly : j - 1)
    for n in 1:cell_list.counts[i, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, i, jj]) count += 1 end
    end

    ii = (i == lx ? 1 : i + 1)
    for n in 1:cell_list.counts[ii, j]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, j]) count += 1 end
    end

    jj = (j == ly ? 1 : j + 1)
    for n in 1:cell_list.counts[i, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, i, jj]) count += 1 end
    end

    return count
end

@inline function count_diagonal_overlaps(colloid::Colloid, cell_list::CuCellList,
                                         i::Integer, j::Integer, idx::Integer)
    count = 0
    lx, ly = size(cell_list.counts)

    ii, jj = (i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)
    for n in 1:cell_list.counts[ii, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, jj]) count += 1 end
    end

    ii, jj = (i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)
    for n in 1:cell_list.counts[ii, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, jj]) count += 1 end
    end

    ii, jj = (i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)
    for n in 1:cell_list.counts[ii, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, jj]) count += 1 end
    end

    ii, jj = (i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)
    for n in 1:cell_list.counts[ii, jj]
        if is_overlapping(colloid, idx, cell_list.cells[n, ii, jj]) count += 1 end
    end

    return count
end
