abstract type CellList end

struct SeqCellList <: CellList
    cells::Matrix{Vector{Int}}
    width::Tuple{<:Real, <:Real}

    function SeqCellList(particles::RegularPolygons)
        d = 2 * particles.radius
        width = (d + (particles.boxsize[1] % d) / (particles.boxsize[1] ÷ d),
                 d + (particles.boxsize[2] % d) / (particles.boxsize[2] ÷ d))

        cells = [Int[] for i in 1:Int(particles.boxsize[1] ÷ width[1]),
                           j in 1:Int(particles.boxsize[2] ÷ width[2])]
        for idx in 1:particlecount(particles)
            i, j = get_cell_list_indices(particles, size(cells), width, idx)
            push!(cells[i, j], idx)
        end

        new(cells, width)
    end
end

@inline function get_cell_list_indices(particles::RegularPolygons,
        gridsize::Tuple{<:Integer, <:Integer}, width::Tuple{<:Real, <:Real}, idx::Integer)
    shearshift = particles.boxsize[1] / 2 - particles.centers[2, idx] * particles.boxshear[]
    i = min(gridsize[1], Int((particles.centers[1, idx] + shearshift) ÷ width[1] + 1))
    j = min(gridsize[2], Int((particles.centers[2, idx] + particles.boxsize[2] / 2)
                             ÷ width[2] + 1))
    return i, j
end

@inline function get_cell_list_indices(particles::RegularPolygons, cell_list::SeqCellList,
                                       idx::Integer)
    get_cell_list_indices(particles, size(cell_list.cells), cell_list.width, idx)
end

function has_overlap(particles::RegularPolygons, cell_list::SeqCellList,
                     idx::Integer, i::Integer, j::Integer)   
    for n in cell_list.cells[i, j]
        if n != idx && is_overlapping(particles, n, idx)
            return true
        end
    end
    if has_orthogonal_overlap(particles, cell_list, i, j, idx)
        return true
    end
    if has_diagonal_overlap(particles, cell_list, i, j, idx)
        return true
    end
    return false
end

function has_overlap(particles::RegularPolygons, cell_list::SeqCellList)
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for m in eachindex(cell_list.cells[cell])
            idx = cell_list.cells[cell][m]
            for n in m+1:length(cell_list.cells[cell])
                if is_overlapping(particles, idx, cell_list.cells[cell][n])
                    return true
                end
            end
            if (i + j) % 2 == 0
                if has_orthogonal_overlap(particles, cell_list, i, j, idx)
                    return true
                end
            end
            if i % 2 == 0
                if has_diagonal_overlap(particles, cell_list, i, j, idx)
                    return true
                end
            end
        end
    end
    return false
end

function count_overlaps(particles::RegularPolygons, cell_list::SeqCellList)
    overlap_count = 0
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for m in eachindex(cell_list.cells[cell])
            idx = cell_list.cells[cell][m]
            for n in m+1:length(cell_list.cells[cell])
                if is_overlapping(particles, idx, cell_list.cells[cell][n])
                    overlap_count += 1
                end
            end
            if (i + j) % 2 == 0
                overlap_count += count_orthogonal_overlaps(particles, cell_list, i, j, idx)
            end
            if i % 2 == 0
                overlap_count += count_diagonal_overlaps(particles, cell_list, i, j, idx)
            end
        end
    end
    return overlap_count
end

@inline function has_orthogonal_overlap(particles::RegularPolygons, cell_list::SeqCellList,
                                        i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), j]
        if is_overlapping(particles, m, n) return true end
    end
    for n in cell_list.cells[i, (j == 1 ? ly : j - 1)]
        if is_overlapping(particles, m, n) return true end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), j]
        if is_overlapping(particles, m, n) return true end
    end
    for n in cell_list.cells[i, (j == ly ? 1 : j + 1)]
        if is_overlapping(particles, m, n) return true end
    end
    return false
end

@inline function has_diagonal_overlap(particles::RegularPolygons, cell_list::SeqCellList,
                                      i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(particles, m, n) return true end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(particles, m, n) return true end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(particles, m, n) return true end
    end
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(particles, m, n) return true end
    end
    return false
end

@inline function count_orthogonal_overlaps(particles::RegularPolygons, cell_list::SeqCellList,
                                           i::Integer, j::Integer, m::Integer)
    count = 0
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), j]
        if is_overlapping(particles, m, n) count += 1 end
    end
    for n in cell_list.cells[i, (j == 1 ? ly : j - 1)]
        if is_overlapping(particles, m, n) count += 1 end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), j]
        if is_overlapping(particles, m, n) count += 1 end
    end
    for n in cell_list.cells[i, (j == ly ? 1 : j + 1)]
        if is_overlapping(particles, m, n) count += 1 end
    end
    return count
end

@inline function count_diagonal_overlaps(particles::RegularPolygons, cell_list::SeqCellList,
                                         i::Integer, j::Integer, m::Integer)
    count = 0
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(particles, m, n) count += 1 end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(particles, m, n) count += 1 end
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(particles, m, n) count += 1 end
    end
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(particles, m, n) count += 1 end
    end
    return count
end

function calculate_potentials!(particles::RegularPolygons, cell_list::SeqCellList,
        potential::Union{Nothing, Function}, pairpotential::Union{Nothing, Function},
        particle_potentials::Vector{<:Real})
    if !isnothing(potential)
        map!(idx -> potential(particles, idx), particle_potentials,
             1:particlecount(particles))
    else
        particle_potentials .= 0
    end
    if !isnothing(pairpotential)
        calculate_pairpotentials!(particles, cell_list, pairpotential, particle_potentials)
    end
end

function calculate_pairpotentials!(particles::RegularPolygons, cell_list::SeqCellList,
        pairpotential::Function, particle_potentials::Vector{<:Real})
    for cell in CartesianIndices(cell_list.cells)
        i, j = Tuple(cell)
        for m in eachindex(cell_list.cells[cell])
            idx = cell_list.cells[cell][m]
            if (i + j) % 2 == 0
                overlap_count += calculate_orthogonal_pairpotentials!(
                    particles, cell_list, pairpotential, particle_potentials, i, j, idx)
            end
            if i % 2 == 0
                overlap_count += calculate_diagonal_pairpotentials!(
                    particles, cell_list, pairpotential, particle_potentials, i, j, idx)
            end
        end
    end
end

@inline function calculate_orthogonal_pairpotentials!(
        particles::RegularPolygons, cell_list::SeqCellList, pairpotential::Function,
        particle_potentials::Vector{<:Real}, i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), j]
        particle_potentials[m] += pairpotential(particles, m, n)
    end
    for n in cell_list.cells[i, (j == 1 ? ly : j - 1)]
        particle_potentials[m] += pairpotential(particles, m, n)
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), j]
        particle_potentials[m] += pairpotential(particles, m, n)
    end
    for n in cell_list.cells[i, (j == ly ? 1 : j + 1)]
        particle_potentials[m] += pairpotential(particles, m, n)
    end
end

@inline function calculate_diagonal_pairpotentials!(
        particles::RegularPolygons, cell_list::SeqCellList, pairpotential::Function,
        particle_potentials::Vector{<:Real}, i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list.cells)
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)]
        particle_potentials[m] += pairpotential(particles, m, n)
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)]
        particle_potentials[m] += pairpotential(particles, m, n)
    end
    for n in cell_list.cells[(i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)]
        particle_potentials[m] += pairpotential(particles, m, n)
    end
    for n in cell_list.cells[(i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)]
        particle_potentials[m] += pairpotential(particles, m, n)
    end
end

@inline function pairpotential_sum(particles::RegularPolygons, cell_list::SeqCellList,
        pairpotential::Function, idx::Integer, i::Integer, j::Integer)
    potsum = zero(particles.centers)
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
            potsum += pairpotential(particles, idx, neighbor)
        end
    end
    return potsum
end
