function potential(poly1::AbstractPolygon, poly2::AbstractPolygon,
        boundary_shift::Function = x -> (0, 0), potfunc::Function = inv)
    dist = (poly1.center[1] - poly2.center[1], poly1.center[2] - poly2.center[2])
    shift = boundary_shift(dist)
    dist = (dist[1] + shift[1], dist[2] + shift[2])
    distnorm = √(dist[1]^2 + dist[2]^2)

    if distnorm > poly1.radius + poly2.radius
        return 0 * distnorm # for correct type
    elseif distnorm <= poly1.bisector + poly2.bisector
        return potfunc(distnorm)
    end

    single_potential = _corner_invpotential(poly1, poly2, shift, potfunc)
    if single_potential == 0
        return _corner_invpotential(poly2, poly1, shift, potfunc)
    else
        return single_potential
    end
end

function _corner_potential(refpoly::AbstractPolygon, testpoly::AbstractPolygon,
        shift::Tuple{Vararg{<:Real}}, potfunc::Function = inv)
    for vertex in eachcol(testpoly.vertices)
        overlap = true
        vertex_dist = undef
        for normal in eachcol(refpoly.normals)
            vertex_dist, vertex_outside_normal = _vertex_dist_calc(
                refpoly, vertex, normal, shift)
            if vertex_outside_normal
                overlap = false
                break
            end
        end
        if overlap
            return potfunc(√(vertex_dist[1]^2 + vertex_dist[2]^2))
        end
    end
    return 0 * refpoly.center
end

@inline function _vertex_dist_calc(refpoly::RegPoly, vertex::AbstractVector{<:Real},
        normal::AbstractVector{<:Real}, shift::Tuple{Vararg{<:Real}})
    distvec = (vertex[1] - refpoly.center[1] + shift[1],
        vertex[2] - refpoly.center[2] + shift[2])
    return distvec, distvec[1] * normal[1] + distvec[2] * normal[2] > refpoly.bisector
end

@inline function _vertex_dist_calc(refpoly::RegEvenPoly, vertex::AbstractVector{<:Real},
        normal::AbstractVector{<:Real})
    distvec = (vertex[1] - refpoly.center[1] + shift[1],
        vertex[2] - refpoly.center[2] + shift[2])
    normaldist = distvec[1] * normal[1] + distvec[2] * normal[2]
    return distvec, normaldist > refpoly.bisector || normaldist < -refpoly.bisector
end
