function potential(poly1::AbstractPolygon, poly2::AbstractPolygon;
        potfunc::Function = inv)
    centerdist = √((poly1.center[1] - poly2.center[1])^2
        + (poly1.center[2] - poly2.center[2])^2)

    if centerdist > poly1.radius + poly2.radius
        return 0 * centerdist # for correct type
    elseif centerdist <= poly1.bisector + poly2.bisector
        return potfunc(centerdist)
    end

    single_potential = _corner_invpotential(poly1, poly2; potfunc=potfunc)
    if single_potential == 0
        return _corner_invpotential(poly2, poly1; potfunc=potfunc)
    else
        return single_potential
    end
end

function potential_periodic(poly1::AbstractPolygon, poly2::AbstractPolygon,
        boxsize::Tuple{Vararg{<:Real}}; potfunc::Function = inv)
    distvec = (poly1.center[1] - poly2.center[1], poly1.center[2] - poly2.center[2])
    distvec = (distvec[1] - (distvec[1] ÷ (boxsize[1] / 2)) .* boxsize[1],
        distvec[2] - (distvec[2] ÷ (boxsize[2] / 2)) .* boxsize[2])
    centerdist = √(distvec[1]^2 + distvec[2]^2)

    if centerdist > poly1.radius + poly2.radius
        return 0 * centerdist # for correct type
    elseif centerdist <= poly1.bisector + poly2.bisector
        return potfunc(centerdist)
    end

    single_potential = _corner_invpotential(poly1, poly2; potfunc=potfunc)
    if single_potential == 0
        return _corner_invpotential(poly2, poly1; potfunc=potfunc)
    else
        return single_potential
    end
end

function _corner_potential(refpoly::AbstractPolygon, testpoly::AbstractPolygon;
        potfunc::Function = inv)
    for vertex in eachcol(testpoly.vertices)
        overlap = true
        vertex_dist = undef
        for normal in eachcol(refpoly.normals)
            vertex_dist, vertex_outside_normal = _vertex_dist_calc(refpoly, vertex, normal)
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

function _corner_potential_periodic(refpoly::AbstractPolygon, testpoly::AbstractPolygon,
        boxsize::Tuple{Vararg{<:Real}}; potfunc::Function = inv)
    for vertex in eachcol(testpoly.vertices)
        overlap = true
        vertex_dist = undef
        for normal in eachcol(refpoly.normals)
            vertex_dist, vertex_outside_normal = _vertex_dist_calc_periodic(
                refpoly, vertex, normal, boxsize)
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
        normal::AbstractVector{<:Real})
    distvec = (vertex[1] - refpoly.center[1], vertex[2] - refpoly.center[2])
    return distvec, distvec[1] * normal[1] + distvec[2] * normal[2] > refpoly.bisector
end

@inline function _vertex_dist_calc(refpoly::RegEvenPoly, vertex::AbstractVector{<:Real},
        normal::AbstractVector{<:Real})
    distvec = (vertex[1] - refpoly.center[1], vertex[2] - refpoly.center[2])
    normaldist = distvec[1] * normal[1] + distvec[2] * normal[2]
    return distvec, normaldist > refpoly.bisector || normaldist < -refpoly.bisector
end

@inline function _vertex_dist_calc_periodic(refpoly::RegPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real},
        boxsize::Tuple{Vararg{<:Real}})
    distvec = (vertex[1] - refpoly.center[1], vertex[2] - refpoly.center[2])
    distvec = (distvec[1] - (distvec[1] ÷ (boxsize[1] / 2)) .* boxsize[1],
        distvec[2] - (distvec[2] ÷ (boxsize[2] / 2)) .* boxsize[2])
    return distvec, distvec[1] * normal[1] + distvec[2] * normal[2] > refpoly.bisector
end

@inline function _vertex_dist_calc_periodic(refpoly::RegEvenPoly,
    vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real},
    boxsize::Tuple{Vararg{<:Real}})
distvec = (vertex[1] - refpoly.center[1], vertex[2] - refpoly.center[2])
distvec = (distvec[1] - (distvec[1] ÷ (boxsize[1] / 2)) .* boxsize[1],
    distvec[2] - (distvec[2] ÷ (boxsize[2] / 2)) .* boxsize[2])
normaldist = distvec[1] * normal[1] + distvec[2] * normal[2]
return distvec, normaldist > refpoly.bisector || normaldist < -refpoly.bisector
