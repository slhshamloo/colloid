"""
    is_overlapping(poly1::AbstractPolygon, poly2::AbstractPolygon)

Check whether or not `poly1` and `poly2` are is_overlapping. The function is optimized
for even polygons.
"""
function is_overlapping(poly1::AbstractPolygon, poly2::AbstractPolygon,
        boundary_shift::Function = x -> (0, 0))
    dist = (poly1.center[1] - poly2.center[1], poly1.center[2] - poly2.center[2])
    shift = boundary_shift(dist)
    dist = (dist[1] + shift[1], dist[2] + shift[2])
    distnorm = √(dist[1]^2 + dist[2]^2)

    if distnorm > poly1.radius + poly2.radius
        return false
    elseif distnorm <= poly1.bisector + poly2.bisector
        return true
    end
    
    return (_is_vertex_overlapping(poly1, poly2, shift)
        || _is_vertex_overlapping(poly2, poly1, shift))
end

function _is_vertex_overlapping(refpoly::AbstractPolygon, testpoly::AbstractPolygon,
        shift::Tuple{Vararg{<:Real}})
    for vertex in eachcol(testpoly.vertices)
        overlap = true
        for normal in eachcol(refpoly.normals)
            if _is_vertex_outside_normal(refpoly, vertex, normal, shift)
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

macro vertexdot(vertex, center, normal, shift)
    quote
        (($(esc(vertex))[1] - $(esc(center))[1] + $(esc(shift))[1]) * $(esc(normal))[1]
        + ($(esc(vertex))[2] - $(esc(center))[2] + $(esc(shift))[2]) * $(esc(normal))[2])
    end
end

@inline function _is_vertex_outside_normal(refpoly::RegPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real},
        shift::Tuple{Vararg{<:Real}})
    return @vertexdot(vertex, refpoly.center, normal, shift) > refpoly.bisector
end

@inline function _is_vertex_outside_normal(refpoly::RegEvenPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real},
        shift::Tuple{Vararg{<:Real}})
    projected_distance = @vertexdot(vertex, refpoly.center, normal, shift)
    return projected_distance > refpoly.bisector || projected_distance < -refpoly.bisector
end
