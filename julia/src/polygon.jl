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
    θ₀ = π / sidenum
    bisector = radius * cos(θ₀)
    center = MVector{2, F}(center)

    θs = (k * θ₀ + angle for k in 1:2:2sidenum-1)
    vertices = MMatrix{2, sidenum, F}(undef)
    @. vertices[1, :] = radius * cos(θs)
    @. vertices[2, :] = radius * sin(θs)
    vertices .+= center

    θs = (k * θ₀ + angle for k in 0:2:2normalcount-2)
    normals = MMatrix{2, normalcount, F}(undef)
    @. normals[1, :] = cos(θs)
    @. normals[2, :] = sin(θs)
    
    return sidenum, F(radius), F(bisector), center, vertices, normals
end

"""
    rotate!(poly::AbstractPolygon, angle::AbstractFloat)

Move an `AbstractPolygon`
"""
@inline function move!(poly::AbstractPolygon, dist::Tuple{Vararg{<:Real}})
    poly.center[1] += dist[1]
    poly.center[2] += dist[2]
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
        temp = v[1] * cos(angle) - v[2] * sin(angle)
        v[2] = v[1] * sin(angle) + v[2] * cos(angle)
        v[1] = temp
    end
end

"""
    apply_periodic_boundary!(poly::AbstractPolygon, boxsize::Tuple)

Apply periodic boundary conditions with the given box dimensions `boxsize`
"""
@inline function apply_periodic_boundary!(poly::AbstractPolygon,
        boxsize::Tuple{Vararg{<:Real}})
    old_center = (poly.center[1], poly.center[2])
    poly.center[1] = (poly.center[1] + boxsize[1]) % boxsize[1]
    poly.center[2] = (poly.center[2] + boxsize[2]) % boxsize[2]
    shift = (poly.center[1] - old_center[1], poly.center[2] - old_center[2])
    poly.vertices .+= shift
end
