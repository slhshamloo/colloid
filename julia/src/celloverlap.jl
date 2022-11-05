function get_cell_list(colloid::Colloid)
    cell_list = [Int[] for i in 1:Int.(colloid.boxsize[1] ÷ (2 * colloid.radius) + 1),
                           j in 1:Int.(colloid.boxsize[2] ÷ (2 * colloid.radius) + 1)]
    for idx in 1:size(colloid.centers, 2)
        push!(cell_list[
                Int((colloid.centers[1, idx] + colloid.boxsize[1] / 2)
                    ÷ (2 * colloid.radius) + 1),
                Int((colloid.centers[2, idx] + colloid.boxsize[2] / 2)
                    ÷ (2 * colloid.radius) + 1)
            ], idx)
    end
    return cell_list
end

function has_overlap(colloid::Colloid)
    cell_list = get_cell_list(colloid)
    for idx in CartesianIndices(cell_list)
        i, j = Tuple(idx)
        for m in eachindex(cell_list[idx])
            for n in m+1:length(cell_list[idx])
                if is_overlapping(colloid, cell_list[idx][m], cell_list[idx][n])
                    return true
                end
            end
            if (i + j) % 2 == 0
                if check_orthogonal_overlap(colloid, cell_list, i, j, cell_list[idx][m])
                    return true
                end
            end
            if i % 2 == 0
                if check_diagonal_overlap(colloid, cell_list, i, j, cell_list[idx][m])
                    return true
                end
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
