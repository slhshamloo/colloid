function is_overlapping(colloid::Colloid, i::Integer, j::Integer)
    dist = (colloid.centers[1, i] - colloid.centers[1, j],
        colloid.centers[2, i] - colloid.centers[2, j])
    dist = (dist[1] - dist[1] ÷ (colloid.boxsize[1]/2) * colloid.boxsize[1],
            dist[2] - dist[2] ÷ (colloid.boxsize[2]/2) * colloid.boxsize[2])

    distnorm = √(dist[1]^2 + dist[2]^2)
    if distnorm <= 2 * colloid.bisector
        return true
    elseif distnorm > 2 * colloid.radius
        return false
    end
    centerangle = sign(dist[2]) * acos(dist[1] / distnorm)

    return (_is_vertex_overlapping(colloid, i, j, distnorm, centerangle)
            || _is_vertex_overlapping(colloid, j, i, distnorm, π + centerangle))
end

function _is_vertex_overlapping(colloid::Colloid, i::Integer, j::Integer,
                                distnorm::Real, centerangle::Real)
    normalangle = _get_periodic_angle(centerangle - colloid.angles[i], colloid.sidenum)
    vertexangle = _get_periodic_angle(
        π - π / colloid.sidenum + centerangle - colloid.angles[j], colloid.sidenum)
    center_projection = abs(distnorm * cos(normalangle))
    radius_projection = colloid.radius * cos(normalangle - vertexangle)
    return center_projection - radius_projection < colloid.bisector
end

@inline function _get_periodic_angle(angle::Real, sidenum::Integer)
    angle %= 2π / sidenum
    return angle - sign(angle) * π / sidenum
end
