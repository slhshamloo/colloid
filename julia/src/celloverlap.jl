function get_cell_list(colloid::Colloid)
    cell_list = [Int[] for i in 1:Int.(colloid.boxsize[1] ÷ (2 * colloid.radius) + 1),
                           j in 1:Int.(colloid.boxsize[2] ÷ (2 * colloid.radius) + 1)]
    for idx in 1:particle_count(colloid)
        push!(cell_list[
                Int((colloid.centers[1, idx] + colloid.boxsize[1] / 2)
                    ÷ (2 * colloid.radius) + 1),
                Int((colloid.centers[2, idx] + colloid.boxsize[2] / 2)
                    ÷ (2 * colloid.radius) + 1)
            ], idx)
    end
    return cell_list
end

function has_overlap(colloid::Colloid, cell_list::Matrix{Vector{Int}},
                     idx::Integer, i::Integer, j::Integer)   
    for n in cell_list[i, j]
        if n != idx && is_overlapping(colloid, n, idx)
            return true
        end
    end
    if check_orthogonal_overlap(colloid, cell_list, i, j, idx)
        return true
    end
    if check_diagonal_overlap(colloid, cell_list, i, j, idx)
        return true
    end
    return false
end

function count_overlaps(colloid::Colloid, cell_list::Matrix{Vector{Int}})
    overlap_count = 0
    for idx in CartesianIndices(cell_list)
        i, j = Tuple(idx)
        for m in eachindex(cell_list[idx])
            for n in m+1:length(cell_list[idx])
                if is_overlapping(colloid, cell_list[idx][m], cell_list[idx][n])
                    overlap_count += 1
                end
            end
            if (i + j) % 2 == 0
                overlap_count += count_orthogonal_overlap(
                    colloid, cell_list, i, j, cell_list[idx][m])
            end
            if i % 2 == 0
                overlap_count += count_diagonal_overlap(
                    colloid, cell_list, i, j, cell_list[idx][m])
            end
        end
    end
    return false
end

@inline function check_orthogonal_overlap(
        colloid::Colloid, cell_list, i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list)
    for n in cell_list[(i == 1 ? lx : i - 1), j]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list[i, (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list[(i == lx ? 1 : i + 1), j]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list[i, (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    return false
end

@inline function check_diagonal_overlap(
        colloid::Colloid, cell_list, i::Integer, j::Integer, m::Integer)
    lx, ly = size(cell_list)
    for n in cell_list[(i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list[(i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list[(i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    for n in cell_list[(i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) return true end
    end
    return false
end

@inline function count_orthogonal_overlap(
        colloid::Colloid, cell_list, i::Integer, j::Integer, m::Integer)
    count = 0
    lx, ly = size(cell_list)
    for n in cell_list[(i == 1 ? lx : i - 1), j]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list[i, (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list[(i == lx ? 1 : i + 1), j]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list[i, (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    return count
end

@inline function count_diagonal_overlap(
        colloid::Colloid, cell_list, i::Integer, j::Integer, m::Integer)
    count = 0
    lx, ly = size(cell_list)
    for n in cell_list[(i == 1 ? lx : i - 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list[(i == lx ? 1 : i + 1), (j == 1 ? ly : j - 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list[(i == lx ? 1 : i + 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    for n in cell_list[(i == 1 ? lx : i - 1), (j == ly ? 1 : j + 1)]
        if is_overlapping(colloid, m, n) count += 1 end
    end
    return count
end
