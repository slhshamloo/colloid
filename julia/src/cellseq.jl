abstract type CellList end

struct SeqCellList <: CellList
    cells::Matrix{Vector{Int}}
    width::Tuple{<:Real, <:Real}

    function SeqCellList(colloid::Colloid)
        d = 2 * colloid.radius
        width = (d + (colloid.boxsize[1] % d) / (colloid.boxsize[1] ÷ d),
                 d + (colloid.boxsize[2] % d) / (colloid.boxsize[2] ÷ d))

        cells = [Int[] for i in 1:Int(colloid.boxsize[1] ÷ width[1]),
                           j in 1:Int(colloid.boxsize[2] ÷ width[2])]
        for idx in 1:particle_count(colloid)
            i, j = get_cell_list_indices(colloid, size(cells), width, idx)
            push!(cells[i, j], idx)
        end

        new(cells, width)
    end
end

@inline function get_cell_list_indices(colloid::Colloid,
        gridsize::Tuple{<:Integer, <:Integer}, width::Tuple{<:Real, <:Real}, idx::Integer)
    i = min(gridsize[1], Int((colloid.centers[1, idx] + colloid.boxsize[1] / 2)
                             ÷ width[1] + 1))
    j = min(gridsize[2], Int((colloid.centers[2, idx] + colloid.boxsize[2] / 2)
                             ÷ width[2] + 1))

    return i, j
end

@inline function get_cell_list_indices(colloid::Colloid, cell_list::SeqCellList,
                                       idx::Integer)
    get_cell_list_indices(colloid, size(cell_list.cells), cell_list.width, idx)
end

function has_overlap(colloid::Colloid, cell_list::SeqCellList,
                     idx::Integer, i::Integer, j::Integer)   
    for n in cell_list.cells[i, j]
        if n != idx && is_overlapping(colloid, n, idx)
            return true
        end
    end
    if has_orthogonal_overlap(colloid, cell_list, i, j, idx)
        return true
    end
    if has_diagonal_overlap(colloid, cell_list, i, j, idx)
        return true
    end
    return false
end

function has_overlap(colloid::Colloid, cell_list::SeqCellList)
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for m in eachindex(cell_list.cells[cell])
            idx = cell_list.cells[cell][m]
            for n in m+1:length(cell_list.cells[cell])
                if is_overlapping(colloid, idx, cell_list.cells[cell][n])
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

function count_overlaps(colloid::Colloid, cell_list::SeqCellList)
    overlap_count = 0
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for m in eachindex(cell_list.cells[cell])
            idx = cell_list.cells[cell][m]
            for n in m+1:length(cell_list.cells[cell])
                if is_overlapping(colloid, idx, cell_list.cells[cell][n])
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

@inline function has_orthogonal_overlap(colloid::Colloid, cell_list::SeqCellList,
                                        i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), j]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list.cells[i, (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), j]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list.cells[i, (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    return false
end

@inline function has_diagonal_overlap(colloid::Colloid, cell_list::SeqCellList,
                                      i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    return false
end

@inline function count_orthogonal_overlaps(colloid::Colloid, cell_list::SeqCellList,
                                           i::Integer, j::Integer, m::Integer)
    count = 0
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), j]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list.cells[i, (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), j]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list.cells[i, (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    return count
end

@inline function count_diagonal_overlaps(colloid::Colloid, cell_list::SeqCellList,
                                         i::Integer, j::Integer, m::Integer)
    count = 0
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    return count
end

function calculate_potentials!(colloid::Colloid, cell_list::SeqCellList,
        potential::Union{Nothing, Function}, pairpotential::Union{Nothing, Function},
        particle_potentials::Vector{<:Real})
    if !isnothing(potential)
        map!(idx -> potential(colloid, idx), particle_potentials, 1:particle_count(colloid))
    else
        particle_potentials .= 0
    end
    if !isnothing(pairpotential)
        calculate_pairpotentials!(colloid, cell_list, pairpotential, particle_potentials)
    end
end

function calculate_pairpotentials!(colloid::Colloid, cell_list::SeqCellList,
        pairpotential::Function, particle_potentials::Vector{<:Real})
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for m in eachindex(cell_list.cells[cell])
            idx = cell_list.cells[cell][m]
            if (i + j) % 2 == 0
                overlap_count += calculate_orthogonal_pairpotentials!(
                    colloid, cell_list, pairpotential, particle_potentials, i, j, idx)
            end
            if i % 2 == 0
                overlap_count += calculate_diagonal_pairpotentials!(
                    colloid, cell_list, pairpotential, particle_potentials, i, j, idx)
            end
        end
    end
end

@inline function calculate_orthogonal_pairpotentials!(
        colloid::Colloid, cell_list::SeqCellList, pairpotential::Function,
        particle_potentials::Vector{<:Real}, i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), j]
        particle_potentials[m] += pairpotential(colloid, m, n)
    end
    for n in cell_list.cells[i, (j == 1 ? ly : j - 1)]
        particle_potentials[m] += pairpotential(colloid, m, n)
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), j]
        particle_potentials[m] += pairpotential(colloid, m, n)
    end
    for n in cell_list.cells[i, (j == ly ? 1 : j + 1)]
        particle_potentials[m] += pairpotential(colloid, m, n)
    end
end

@inline function calculate_diagonal_pairpotentials!(
        colloid::Colloid, cell_list::SeqCellList, pairpotential::Function,
        particle_potentials::Vector{<:Real}, i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)]
        particle_potentials[m] += pairpotential(colloid, m, n)
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)]
        particle_potentials[m] += pairpotential(colloid, m, n)
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)]
        particle_potentials[m] += pairpotential(colloid, m, n)
    end
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)]
        particle_potentials[m] += pairpotential(colloid, m, n)
    end
end

@inline function pairpotential_sum(colloid::Colloid, cell_list::SeqCellList,
        pairpotential::Function, idx::Integer, i::Integer, j::Integer)
    potsum = zero(colloid.centers)
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
            potsum += pairpotential(colloid, idx, neighbor)
        end
    end
    return potsum
end
