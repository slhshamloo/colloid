function get_cell_list(colloid::Colloid)
    d = 2 * colloid.radius
    cell_width = (d + (colloid.boxsize[1] % d) / (colloid.boxsize[1] ÷ d),
                  d + (colloid.boxsize[2] % d) / (colloid.boxsize[2] ÷ d))
    cell_list = [Int[] for i in 1:Int(colloid.boxsize[1] ÷ cell_width[1]),
                           j in 1:Int(colloid.boxsize[2] ÷ cell_width[2])]
    for idx in 1:particle_count(colloid)
        push!(cell_list[
            min(size(cell_list, 1), Int((colloid.centers[1, idx] + colloid.boxsize[1] / 2)
                                        ÷ cell_width[1] + 1)),
            min(size(cell_list, 2), Int((colloid.centers[2, idx] + colloid.boxsize[2] / 2)
                                        ÷ cell_width[2] + 1))
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
    if has_orthogonal_overlap(colloid, cell_list, i, j, idx)
        return true
    end
    if has_diagonal_overlap(colloid, cell_list, i, j, idx)
        return true
    end
    return false
end

function has_overlap(colloid::Colloid, cell_list::Matrix{Vector{Int}})
    for idx in CartesianIndices(cell_list)
        i, j = Tuple(idx)
        for m in eachindex(cell_list[idx])
            for n in m+1:length(cell_list[idx])
                if is_overlapping(colloid, cell_list[idx][m], cell_list[idx][n])
                    return true
                end
            end
            if (i + j) % 2 == 0
                if has_orthogonal_overlap(colloid, cell_list, i, j, cell_list[idx][m])
                    return true
                end
            end
            if i % 2 == 0
                if has_diagonal_overlap(colloid, cell_list, i, j, cell_list[idx][m])
                    return true
                end
            end
        end
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
                overlap_count += count_orthogonal_overlaps(
                    colloid, cell_list, i, j, cell_list[idx][m])
            end
            if i % 2 == 0
                overlap_count += count_diagonal_overlaps(
                    colloid, cell_list, i, j, cell_list[idx][m])
            end
        end
    end
    return overlap_count
end

@inline function has_orthogonal_overlap(
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

@inline function has_diagonal_overlap(
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

@inline function count_orthogonal_overlaps(
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

@inline function count_diagonal_overlaps(
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
