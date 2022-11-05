function is_overlapping(colloid::Colloid, i::Integer, j::Integer)
    dist = (colloid.centers[1, i] - colloid.centers[1, j],
            colloid.centers[2, i] - colloid.centers[2, j])

    if dist[1] > colloid.boxsize[1] / 2 || dist[2] > colloid.boxsize[2] / 2
        shift = (-(dist[1] ÷ (colloid.boxsize[1]/2)) * colloid.boxsize[1],
                 -(dist[2] ÷ (colloid.boxsize[2]/2)) * colloid.boxsize[2])
        dist = (dist[1] + shift[1], dist[2] + shift[2])
        is_vertex_outside_normal = (colloid.sidenum % 2 == 0 ?
            (c, ii, jj, v, n) -> _is_vertex_outside_normal_even_p(c, ii, jj, v, n, shift)
            : (c, ii, jj, v, n) -> _is_vertex_outside_normal_odd_p(c, ii, jj, v, n, shift))
    else
        is_vertex_outside_normal = (colloid.sidenum % 2 == 0 ?
            _is_vertex_outside_normal_even : _is_vertex_outside_normal_odd)
    end

    dist_norm_squared = dist[1]^2 + dist[2]^2
    if dist_norm_squared <= 4 * colloid.bisector^2
        return true
    elseif dist_norm_squared > 4 * colloid.radius^2
        return false
    end

    return (_is_vertex_overlapping(colloid, i, j, is_vertex_outside_normal)
        || _is_vertex_overlapping(colloid, j, i, is_vertex_outside_normal))
end

function _is_vertex_overlapping(colloid::Colloid, i::Integer, j::Integer,
        is_vertex_outside_normal::Function)
    for v in 1:colloid.sidenum
        overlap = true
        for n in 1:size(colloid.normals, 2)
            if is_vertex_outside_normal(colloid, i, j, v, n)
                overlap = false
                break
            end
        end
        if overlap
            return true
        end
    end
    return false
end

@inline function _is_vertex_outside_normal_even(
        colloid::Colloid, i::Integer, j::Integer, v::Integer, n::Integer)
    projection = (
        (colloid.vertices[1, v, i] - colloid.centers[1, j]) * colloid.normals[1, n, j]
        + (colloid.vertices[2, v, i] - colloid.centers[2, j]) * colloid.normals[2, n, j])
    return projection > colloid.bisector || projection < -colloid.bisector
end

@inline function _is_vertex_outside_normal_odd(
        colloid::Colloid, i::Integer, j::Integer, v::Integer, n::Integer)
    projection = (
        (colloid.vertices[1, v, i] - colloid.centers[1, j]) * colloid.normals[1, n, j]
        + (colloid.vertices[2, v, i] - colloid.centers[2, j]) * colloid.normals[2, n, j])
    return projection > colloid.bisector
end

@inline function _is_vertex_outside_normal_even_p(
        colloid::Colloid, i::Integer, j::Integer, v::Integer, n::Integer,
        shift::Tuple{Vararg{<:Real}})
    projection = (
        (colloid.vertices[1, v, i] - colloid.centers[1, j] + shift[1])
            * colloid.normals[1, n, j]
        + (colloid.vertices[2, v, i] - colloid.centers[2, j] + shift[2])
            * colloid.normals[2, n, j])
    return projection > colloid.bisector || projection < -colloid.bisector
end

@inline function _is_vertex_outside_normal_odd_p(
        colloid::Colloid, i::Integer, j::Integer, v::Integer, n::Integer,
        shift::Tuple{Vararg{<:Real}})
    projection = (
        (colloid.vertices[1, v, i] - colloid.centers[1, j] + shift[1])
            * colloid.normals[1, n, j]
        + (colloid.vertices[2, v, i] - colloid.centers[2, j] + shift[2])
            * colloid.normals[2, n, j])
    return projection > colloid.bisector
end
