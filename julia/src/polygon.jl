abstract type AbstractPolygon end

"""
    RegPoly{F}(sidenum, radius, angle, center)
    RegPoly(sidenum, radius, angle, center)

A regular polygon with `sidenum` sides, side length `sidelen`, counter-clockwise rotation
about its center `angle`, and center coordinates `center`; first, one side is set to be
prependicular to the x axis (the first axis), and the rotation angle is defined relative to
this configuration. `F` is the floating point type. If not specified, it is set to
`Float32`.
"""
struct RegPoly{F<:AbstractFloat} <: AbstractPolygon
    sidenum::Integer
    radius::F
    bisector::F
    center::AbstractVector
    vertices::AbstractMatrix
    normals::AbstractMatrix

    function RegPoly{F}(sidenum::Integer, radius::Real, angle::Real,
            center::Tuple{Vararg{<:Real}}) where {F<:AbstractFloat}
        new{F}(_build_regpoly_attributes(F, sidenum, sidenum, radius, angle, center)...)
    end
end

function RegPoly(sidenum::Integer, radius::Real, angle::Real,
        center::Tuple{Vararg{<:Real}})
    RegPoly{Float32}(sidenum, radius, angle, center)
end

"""
    RegEvenPoly{F}(sidenum, radius, angle, center)
    RegEvenPoly(sidenum, radius, angle, center)

Like `RegPoly`, but optimized for even-sided polygons; only half of the normals are stored
as the other half are just mirrored versions of the stored half.
"""
struct RegEvenPoly{F<:AbstractFloat} <: AbstractPolygon
    sidenum::Integer
    radius::F
    bisector::F
    center::AbstractVector
    vertices::AbstractMatrix
    normals::AbstractMatrix

    function RegEvenPoly{F}(sidenum::Integer, radius::Real, angle::Real,
            center::Tuple{Vararg{<:Real}}) where {F<:AbstractFloat}
        new{F}(_build_regpoly_attributes(F, sidenum, sidenum÷2, radius, angle, center)...)
    end
end

function RegEvenPoly(sidenum::Integer, radius::Real, angle::Real,
        center::Tuple{Vararg{<:Real}})
    RegEvenPoly{Float32}(sidenum, radius, angle, center)
end

@inline function _build_regpoly_attributes(F, sidenum::Integer, normalcount::Integer,
        radius::Real, angle::Real, center::Tuple{Vararg{Real}})
    θ₀ = F(π / sidenum)
    bisector = F(radius * cos(θ₀))
    center = MVector{2}(F.(center))

    θs = (k * θ₀ + angle for k in 1:2:2sidenum-1)
    vertices = MMatrix{2, sidenum, F}(undef)
    @. vertices[1, :] = radius * cos(θs)
    @. vertices[2, :] = radius * sin(θs)
    vertices .+= center

    θs = (k * θ₀ + angle for k in 0:2:2normalcount-2)
    normals = MMatrix{2, normalcount, F}(undef)
    @. normals[1, :] = cos(θs)
    @. normals[2, :] = sin(θs)
    
    return sidenum, radius, bisector, center, vertices, normals
end

"""
    rotate!(poly::AbstractPolygon, angle::AbstractFloat)

Move an `AbstractPolygon`
"""
@inline function move!(poly::AbstractPolygon, dist::Tuple{Vararg{<:Real}})
    poly.center .+= dist
    poly.vertices .+= dist
end

"""
    rotate!(poly::AbstractPolygon, angle::AbstractFloat)

Rotate an `AbstractPolygon`
"""
@inline function rotate!(poly::AbstractPolygon, angle::Real)
    poly.vertices .-= poly.center
    rotate2d!(poly.vertices, angle)
    poly.vertices .+= poly.center

    rotate2d!(poly.normals, angle)
end

@inline function rotate2d!(vectors::AbstractVecOrMat, angle::Real)
    for v in eachcol(vectors)
        v[1] = v[1] * cos(angle) - v[2] * sin(angle)
        v[2] = v[1] * sin(angle) + v[2] * cos(angle)
    end
end

"""
    apply_periodic_boundary!(poly::AbstractPolygon, boxsize::Tuple)

Apply periodic boundary conditions with the given box dimensions `boxsize`
"""
@inline function apply_periodic_boundary!(poly::AbstractPolygon,
        boxsize::Tuple{Vararg{<:Real}})
    poly.center .= (poly.center .+ boxsize) .% boxsize
    poly.vertices .= (poly.vertices .+ boxsize) .% boxsize
end

"""
    is_overlapping(poly1::AbstractPolygon, poly2::AbstractPolygon)

Check whether or not `poly1` and `poly2` are is_overlapping. The function is optimized
for even polygons.
"""
function is_overlapping(poly1::AbstractPolygon, poly2::AbstractPolygon)
    centerdist = √((poly1.center[1] - poly2.center[1])^2
        + (poly1.center[2] - poly2.center[2])^2)
    if centerdist > poly1.radius + poly2.radius
        return false
    elseif centerdist <= poly1.bisector + poly2.bisector
        return true
    end
    
    return _is_vertex_overlapping(poly1, poly2) || _is_vertex_overlapping(poly2, poly1)
end

"""
    is_overlapping_periodic(poly1::AbstractPolygon, poly2::AbstractPolygon,
        boxsize::Tuple{Vararg{<:Real}})

Like `is_overlapping`, but with periodic boundary conditions.
"""
function is_overlapping_periodic(poly1::AbstractPolygon, poly2::AbstractPolygon,
        boxsize::Tuple{Vararg{<:Real}})
    distvec = (poly1.center[1] - poly2.center[1], poly1.center[2] - poly2.center[2])
    distvec = (distvec[1] - (distvec[1] ÷ (boxsize[1] / 2)) .* boxsize[1],
        distvec[2] - (distvec[2] ÷ (boxsize[2] / 2)) .* boxsize[2])
    centerdist = √(distvec[1]^2 + distvec[2]^2)

    if centerdist > poly1.radius + poly2.radius
        return false
    elseif centerdist <= poly1.bisector + poly2.bisector
        return true
    end

    return (_is_vertex_overlapping_periodic(poly1, poly2, boxsize)
        || _is_vertex_overlapping_periodic(poly2, poly1, boxsize))
end

function _is_vertex_overlapping(refpoly::AbstractPolygon, testpoly::AbstractPolygon)
    for vertex in eachcol(testpoly.vertices)
        overlap = true
        for normal in eachcol(refpoly.normals)
            if _is_vertex_outside_normal(refpoly, vertex, normal)
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

function _is_vertex_overlapping_periodic(refpoly::AbstractPolygon,
        testpoly::AbstractPolygon, boxsize::Tuple{Vararg{<:Real}})
    for vertex in eachcol(testpoly.vertices)
        overlap = true
        for normal in eachcol(refpoly.normals)
            if _is_vertex_outside_normal_periodic(refpoly, vertex, normal, boxsize)
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

macro vertexdot(vertex, center, normal)
    quote
        (($(esc(vertex))[1] - $(esc(center))[1]) * $(esc(normal))[1]
        + ($(esc(vertex))[2] - $(esc(center))[2]) * $(esc(normal))[2])
    end
end

@inline function _is_vertex_outside_normal(refpoly::RegPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real})
    return @vertexdot(vertex, refpoly.center, normal) > refpoly.bisector
end

@inline function _is_vertex_outside_normal(refpoly::RegEvenPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real})
    normaldist = @vertexdot(vertex, refpoly.center, normal)
    return normaldist > refpoly.bisector || normaldist < -refpoly.bisector
end

@inline function _is_vertex_outside_normal_periodic(refpoly::RegPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real},
        boxsize::Tuple{Vararg{<:Real}})
    distvec = (vertex[1] - refpoly.center[1], vertex[2] - refpoly.center[2])
    distvec = (distvec[1] - (distvec[1] ÷ (boxsize[1] / 2)) .* boxsize[1],
        distvec[2] - (distvec[2] ÷ (boxsize[2] / 2)) .* boxsize[2])
    return distvec[1] * normal[1] + distvec[2] * normal[2] > refpoly.bisector
end

@inline function _is_vertex_outside_normal_periodic(refpoly::RegEvenPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real},
        boxsize::Tuple{Vararg{<:Real}})
    distvec = (vertex[1] - refpoly.center[1], vertex[2] - refpoly.center[2])
    distvec = (distvec[1] - (distvec[1] ÷ (boxsize[1] / 2)) .* boxsize[1],
        distvec[2] - (distvec[2] ÷ (boxsize[2] / 2)) .* boxsize[2])
    normaldist = distvec[1] * normal[1] + distvec[2] * normal[2]
    return normaldist > refpoly.bisector || normaldist < -refpoly.bisector
end
