function is_overlapping(colloid::Colloid, i::Integer, j::Integer)
    definite_overlap, out_of_range, dist, distnorm = _overlap_range(colloid, i, j)
    if definite_overlap
        return true
    end
    if out_of_range
        return false
    end

    centerangle = (dist[2] < 0 ? -1 : 1) * acos(dist[1] / distnorm)
    return (_is_vertex_overlapping(colloid, i, j, distnorm, centerangle)
            || _is_vertex_overlapping(colloid, j, i, distnorm, π + centerangle))
end

@inline function _overlap_range(colloid::Colloid, i::Integer, j::Integer)
    dist = (colloid.centers[1, i] - colloid.centers[1, j],
            colloid.centers[2, i] - colloid.centers[2, j])
    dist = (dist[1] - dist[1] ÷ (colloid.boxsize[1]/2) * colloid.boxsize[1],
            dist[2] - dist[2] ÷ (colloid.boxsize[2]/2) * colloid.boxsize[2])
    distnorm = √(dist[1]^2 + dist[2]^2)

    return (distnorm <= 2 * colloid.bisector, distnorm > 2 * colloid.radius,
            dist, distnorm)
end

function _is_vertex_overlapping(colloid::Colloid, i::Integer, j::Integer,
                                distnorm::Real, centerangle::Real)
    normalangle = _get_periodic_angle(centerangle - colloid.angles[i], colloid.sidenum)
    vertexangle = _get_periodic_angle(
        π - π / colloid.sidenum + centerangle - colloid.angles[j], colloid.sidenum)
    diffangle = abs(normalangle - vertexangle)

    return (
        (abs(distnorm * cos(normalangle))
            - colloid.radius * cos(diffangle) <= colloid.bisector)
        && (abs(distnorm * cos(2π / colloid.sidenum - abs(normalangle)))
            - colloid.radius * cos(2π / colloid.sidenum - diffangle) <= colloid.bisector)
    )
end

@inline function _get_periodic_angle(angle::Real, sidenum::Integer)
    angle %= 2π / sidenum
    return angle + (angle > 0 ? -1 : 1) * π / sidenum
end

function is_overlapping_with_disk(colloid::Colloid, index::Integer,
        center::Tuple{<:Real, <:Real}, radius::Real)
    definite_overlap, out_of_range, dist, distnorm = _overlap_range_disk(
        colloid, index, center, radius)
    if definite_overlap
        return true
    end
    if out_of_range
        return false
    end
    return _is_disk_over_side(colloid, index, dist, distnorm, radius)
end

@inline function _overlap_range_disk(colloid::Colloid, index::Integer,
        center::Tuple{<:Real, <:Real}, radius::Real)
    dist = (center[1] - colloid.centers[1, index],
            center[2] - colloid.centers[2, index])
    dist = (dist[1] - dist[1] ÷ (colloid.boxsize[1]/2) * colloid.boxsize[1],
            dist[2] - dist[2] ÷ (colloid.boxsize[2]/2) * colloid.boxsize[2])
    distnorm = √(dist[1]^2 + dist[2]^2)
    return (distnorm <= radius + colloid.bisector, distnorm > radius + colloid.radius,
            dist, distnorm)
end

function _is_disk_over_side(colloid::Colloid, index::Integer,
        dist::Tuple{<:Real, <:Real}, distnorm::Real, radius::Real)
    centerangle = (dist[2] < 0 ? -1 : 1) * acos(dist[1] / distnorm)
    colloid_angle = 2π / colloid.sidenum
    vertexnum = fld(centerangle - colloid.angles[index], colloid_angle)
    
    v1 = (colloid.radius * cos(vertexnum * colloid_angle - colloid.angles[index]),
          colloid.radius * sin(vertexnum * colloid_angle - colloid.angles[index]))
    v2 = (colloid.radius * cos((vertexnum + 1) * colloid_angle - colloid.angles[index]),
          colloid.radius * sin((vertexnum + 1) * colloid_angle - colloid.angles[index]))
    
    r1 = (dist[1] - v1[1], dist[2] - v1[2])
    v12 = (v2[1] - v1[1], v2[2] - v1[2])

    v12norm = √(v12[1]^2 + v12[2]^2)
    cross_product = r1[1] * v12[2] - r1[2] * v12[1]
    normal_distance = cross_product / v12norm

    if normal_distance < radius
        r1dot = r1[1] * v12[1] + r1[2] * v12[2]
        if r1dot >= 0
            r2 = (dist[1] - v2[1], dist[2] - v2[2])
            r2dot = r2[1] * v12[1] + r2[2] * v12[2]
            if r2dot <= 0
                return true
            else
                return r1[1]^2 + r1[2]^2 <= radius^2
            end
        else
            return r1[1]^2 + r1[2]^2 <= radius^2
        end
    else
        return false
    end
end
