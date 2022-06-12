abstract type AbstractPolygon end

"""
    RegPoly(sidenum, radius, angle, center)

Create a regular polygon with `sidenum` sides, side length `sidelen`, counter-clockwise
rotation about its center `angle`, and center coordinates `center`; first, one side is set to
be prependicular to the x axis (the first axis), and the rotation angle is defined relative to
this configuration.
"""
struct RegPoly{F<:AbstractFloat} <: AbstractPolygon
    sidenum::Integer
    radius::F
    bisector::F
    center::AbstractVector
    vertices::AbstractMatrix
    normals::AbstractMatrix

    function RegPoly{F}(sidenum::Integer, radius::Real, angle::Real,
            center::AbstractVector{<:Real}) where {F<:AbstractFloat}
        bisector, center, vertices, θs, θ₀ = _build_regpoly_attributes{F}(
            sidenum, radius, angle, center)

        θs .-= θ₀
        normals = MMatrix{2, sidenum, F}(vcat(cos.(θs)', sin.(θs)'))

        new{F}(sidenum, radius, bisector, center, vertices, normals)
    end
end

function RegPoly(sidenum::Integer, radius::Real, angle::Real,
        center::AbstractVector{<:Real})
    RegPoly{Float32}(sidenum, radius, angle, center)
end

"""
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
            center::AbstractVector{<:Real}) where {F<:AbstractFloat}
        bisector, center, vertices, θs, θ₀ = _build_regpoly_attributes{F}(
            sidenum, radius, angle, center)

        θs = θs[1:sidenum÷2] .- θ₀
        normals = MMatrix{2, sidenum÷2, F}(vcat(cos.(θs)', sin.(θs)'))

        new{F}(sidenum, radius, bisector, center, vertices, normals)
    end
end

function RegEvenPoly(sidenum::Integer, radius::Real, angle::Real,
        center::AbstractVector{<:Real})
    RegEvenPoly{Float32}(sidenum, radius, angle, center)
end

@inline function _build_regpoly_attributes{F}(sidenum::Integer, radius::Real, angle::Real,
        center::AbstractVector{<:Real}) where {F<:AbstractFloat}
    θ₀ = F(π / sidenum)
    bisector = F(radius * cos(θ₀))
    center = MVector{2}(F.(center))

    θs = MVector{sidenum}([(2i + 1) * θ₀ + angle for i in 0:sidenum-1])
    vertices = MMatrix{2, sidenum, F}(radius * vcat(cos.(θs)', sin.(θs)'))
    vertices .+= center
    
    return bisector, center, vertices, θs, θ₀
end

"""
    rotate!(poly::AbstractPolygon, angle::AbstractFloat)

Move an `AbstractPolygon`
"""
@inline function move!(poly::AbstractPolygon, dist::AbstractVector{<:AbstractFloat})
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

@inline function apply_periodic_boundary!(poly::AbstractPolygon,
        boxsize::Tuple{<:Real, <:Real})
    
end

"""
    is_overlapping(poly1::AbstractPolygon, poly2::AbstractPolygon)

Check whether or not `poly1` and `poly2` are is_overlapping. The function is optimized
for even polygons.
"""
function is_overlapping(poly1::AbstractPolygon, poly2::AbstractPolygon)
    centerdist = √sum(^(2), poly1.center - poly2.center)
    if centerdist > poly1.radius + poly2.radius
        return false
    elseif centerdist <= poly1.bisector + poly2.bisector
        return true
    end
    
    return _is_vertex_overlapping(poly1, poly2) || _is_vertex_overlapping(poly2, poly1)
end

"""
    is_overlapping_periodic(poly1::AbstractPolygon, poly2::AbstractPolygon,
        boxsize::Tuple{<:Real, <:Real})

Like `is_overlapping`, but for periodic boundary conditions.
"""
function is_overlapping_periodic(poly1::AbstractPolygon, poly2::AbstractPolygon,
        boxsize::Tuple{<:Real, <:Real})
    distvec = (poly1.center - poly2.center)
    distvec -= (dist .÷ (boxsize ./ 2)) .* boxsize
    centerdist = √sum(^(2), distvec)

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
        testpoly::AbstractPolygon, boxsize::Tuple{<:Real, <:Real})
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

@inline function _is_vertex_outside_normal(refpoly::RegPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real})
    return sum((vertex - center) .* normal) > refpoly.bisector
end

@inline function _is_vertex_outside_normal(refpoly::RegEvenPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real})
    normaldist = sum((vertex - center) .* normal)
    return normaldist > refpoly.bisector || normaldist < -refpoly.bisector
end

@inline function _is_vertex_outside_normal_periodic(refpoly::RegPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real},
        boxsize::Tuple{<:Real, <:Real})
    distvec = vertex - center
    distvec -= (dist .÷ (boxsize ./ 2)) .* boxsize
    return sum(distvec .* normal) > refpoly.bisector
end

@inline function _is_vertex_outside_normal_periodic(refpoly::RegEvenPoly,
        vertex::AbstractVector{<:Real}, normal::AbstractVector{<:Real},
        boxsize::Tuple{<:Real, <:Real})
    distvec = vertex - center
    distvec -= (dist .÷ (boxsize ./ 2)) .* boxsize
    normaldist = sum(distvec .* normal)
    return normaldist > refpoly.bisector || normaldist < -refpoly.bisector
end
