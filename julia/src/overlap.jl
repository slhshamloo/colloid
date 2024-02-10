function is_overlapping(particles::RegularPolygons, i::Integer, j::Integer)
    definite_overlap, out_of_range, dist, distnorm = _overlap_range(particles, i, j)
    if definite_overlap
        return true
    end
    if out_of_range
        return false
    end

    centerangle = (dist[2] < 0 ? -1 : 1) * acos(dist[1] / distnorm)
    return (_is_vertex_overlapping(particles, i, j, distnorm, centerangle)
            || _is_vertex_overlapping(particles, j, i, distnorm, π + centerangle))
end

@inline function _overlap_range(particles::RegularPolygons, i::Integer, j::Integer)
    dist = apply_parallelogram_boundary(particles,
        (particles.centers[1, j] - particles.centers[1, i],
         particles.centers[2, j] - particles.centers[2, i]))
    distnorm = √(dist[1]^2 + dist[2]^2)
    return (distnorm <= 2 * particles.bisector, distnorm > 2 * particles.radius,
            dist, distnorm)
end

function _is_vertex_overlapping(particles::RegularPolygons, i::Integer, j::Integer,
                                distnorm::Real, centerangle::Real)
    normalangle = _get_periodic_angle(centerangle - particles.angles[i], particles.sidenum)
    vertexangle = _get_periodic_angle(
        π - π / particles.sidenum + centerangle - particles.angles[j], particles.sidenum)
    diffangle = abs(normalangle - vertexangle)

    return (
        (abs(distnorm * cos(normalangle))
            - particles.radius * cos(diffangle) <= particles.bisector)
        && (abs(distnorm * cos(2π / particles.sidenum - abs(normalangle)))
            - abs(particles.radius * cos(2π / particles.sidenum - diffangle))
            <= particles.bisector)
    )
end

@inline function _get_periodic_angle(angle::Real, sidenum::Integer)
    angle %= 2π / sidenum
    return angle + (angle > 0 ? -1 : 1) * π / sidenum
end

function is_overlapping_with_disk(particles::RegularPolygons, index::Integer,
        center::Tuple{<:Real, <:Real}, radius::Real)
    definite_overlap, out_of_range, dist, distnorm = _overlap_range_disk(
        particles, index, center, radius)
    if definite_overlap
        return true
    end
    if out_of_range
        return false
    end
    return _is_disk_over_side(particles, index, dist, distnorm, radius)
end

@inline function _overlap_range_disk(particles::RegularPolygons, index::Integer,
        center::Tuple{<:Real, <:Real}, radius::Real)
    dist = apply_parallelogram_boundary(particles,
        (center[1] - particles.centers[1, index], center[2] - particles.centers[2, index]))
    distnorm = √(dist[1]^2 + dist[2]^2)
    return (distnorm <= radius + particles.bisector, distnorm > radius + particles.radius,
            dist, distnorm)
end

function _is_disk_over_side(particles::RegularPolygons, index::Integer,
        dist::Tuple{<:Real, <:Real}, distnorm::Real, radius::Real)
    centerangle = (dist[2] < 0 ? -1 : 1) * acos(dist[1] / distnorm)
    particles_angle = 2π / particles.sidenum
    vertexnum = fld(centerangle - particles.angles[index], particles_angle)
    
    v1 = (particles.radius * cos(vertexnum * particles_angle - particles.angles[index]),
          particles.radius * sin(vertexnum * particles_angle - particles.angles[index]))
    v2 = (particles.radius * cos((vertexnum + 1) * particles_angle - particles.angles[index]),
          particles.radius * sin((vertexnum + 1) * particles_angle - particles.angles[index]))
    
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
