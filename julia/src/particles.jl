abstract type ParticleCollection end
"""
    ParticleCollection

Abstract type for particle collections that contain particle and box information.
"""

"""
    RegularPolygons <: ParticleCollection

A collection of regular polygons in a box.

To speed up calculations, the structure contains the `bisector` length.
"""
struct RegularPolygons{T<:Real, V<:AbstractVector, M<:AbstractMatrix,
                       VB<:AbstractVector, VS<:AbstractVector} <: ParticleCollection
    sidenum::Int
    radius::T
    bisector::T

    boxsize::VB
    boxshear::VS

    centers::M
    angles::V
end

"""
    RegularPolygons{T}(sidenum, radius, boxsize::Tuple{<:Real, <:Real}, centers, angels; boxshear::Real = 0, gpu=false) where {T<:Real}
    RegularPolygons{T}(sidenum, radius, boxsize::Tuple{<:Real, <:Real}, count; boxshear::Real = 0, gpu=false) where {T<:Real}

Make a collection of regular polygon particles with `sidenum` sides in a box.

The first method makes the particle collection with `centers` (center coordinates in a
matrix with column-major order) and `angles` of the particles specified. The second method
makes a particle collection with `count` particles with `centers` and `angles` arrays
initialized by `undef`.

`boxshear` is the tangent of the shear angle of the box, defined as the complement of the
complement of the accute angle of the box parallelogram
"""
function RegularPolygons{T}(sidenum::Integer, radius::Real, boxsize::Tuple{<:Real, <:Real},
        centers::AbstractMatrix, angles::AbstractArray;
        boxshear::Real = 0, gpu::Bool = false) where {T<:Real}
    bisector = radius * cos(π / sidenum)

    if gpu
        angles = CuVector{T}(angles)
        centers = CuMatrix{T}(centers)
        boxsize = CuVector{T}([boxsize[1], boxsize[2]])
        boxshear = CuVector{T}([boxshear])
    else
        angles = Vector{T}(angles)
        centers = Matrix{T}(centers)
        boxsize = MVector{2, T}(boxsize)
        boxshear = MVector{1, T}(boxshear)
    end

    RegularPolygons{T, typeof(angles), typeof(centers), typeof(boxsize), typeof(boxshear)}(
        sidenum, radius, bisector, boxsize, boxshear, centers, angles)
end

function RegularPolygons{T}(sidenum::Integer, radius::Real,
        boxsize::Tuple{<:Real, <:Real}, count::Integer;
        gpu::Bool = false, boxshear::Real = 0.0) where {T<:Real}
    bisector = radius * cos(π / sidenum)

    if gpu
        angles = CuVector{T}(undef, count)
        centers = CuMatrix{T}(undef, 2, count)
        boxsize = CuVector{T}([boxsize[1], boxsize[2]])
        boxshear = CuVector{T}([boxshear])
    else
        angles = Vector{T}(undef, count)
        centers = Matrix{T}(undef, 2, count)
        boxsize = MVector{2, T}(boxsize)
        boxshear = MVector{1, T}(boxshear)
    end

    RegularPolygons{T, typeof(angles), typeof(centers), typeof(boxsize), typeof(boxshear)}(
        sidenum, radius, bisector, boxsize, boxshear, centers, angles)
end

Adapt.@adapt_structure RegularPolygons

"""
    particlecount(particles)

Get the number of particles in the particle collection
"""
@inline particlecount(particles::RegularPolygons) = size(particles.centers, 2)

"""
    particlecount(particles)

Calculate the area (2D volume) of the particles in the particle collection.
"""
@inline particlearea(particles::RegularPolygons) = (
    0.5 * particles.sidenum * particles.radius^2 * sin(2π / particles.sidenum))

@inline boxarea(particles::RegularPolygons) =
    CUDA.@allowscalar particles.boxsize[1] * particles.boxsize[2]

function _build_vertices(sidenum::Integer, radius::Real,
        centers::AbstractMatrix, angles::AbstractVector)
    vertices = Array{eltype(centers), 3}(undef, 2, sidenum, size(centers, 2))
    for particle in eachindex(angles)
        θs = (k * π / sidenum + angles[particle] for k in 0:2:2sidenum-2)
        @. vertices[1, :, particle] = radius * cos(θs)
        @. vertices[2, :, particle] = radius * sin(θs)
        vertices[:, :, particle] .+= centers[:, particle]
    end
    return vertices
end

"""
    cystallize!(particles[, gridcount, constaint::Function])

Arrange the particles in a grid spanning the full box.

The number of points of the grid is the closest square number to `gridcount` which is also
larger than it. The grid is filled starting from the shorter side of the box. Constraints
for the position of the particles can be specified by the `constraint` function, which takes
the particle collection as the first arguments and the column-major `centers` matrix of the
particle positions as the second arguments and returns a boolean vector that identifies
which of the coordinate pairs stored in each column of `centers` is valid.
"""
function crystallize!(particles::RegularPolygons,
        gridcount::Integer = particlecount(particles),
        constraint::Function = (particles, centers) -> trues(size(centers, 2)))
    particles_per_side = ceil(Int, sqrt(gridcount))
    shortside = minimum(particles.boxsize)
    shortdim = argmin(particles.boxsize)
    spacing = shortside / particles_per_side

    repvals = range((-shortside + spacing) / 2, (shortside - spacing) / 2,
                    particles_per_side)
    shortpos = repeat(repvals, particles_per_side)
    longpos = repeat(repvals, inner=particles_per_side)

    xs = shortdim == 1 ? shortpos : longpos
    ys = shortdim == 2 ? shortpos : longpos
    centers = permutedims(hcat(xs, ys))
    valid = constraint(particles, centers)
    centers = centers[vcat(valid, valid)]
    centers = reshape(centers, 2, length(centers)÷2)
    centers = centers[:, 1:particlecount(particles)]

    CUDA.@allowscalar(centers[1, :] .+= centers[2, :] * particles.boxshear[])

    if isa(particles.centers, CuArray)
        centers = CuArray(centers)
    end
    particles.centers .= centers
    particles.angles .= 0
end

"""
    move!(particles, idx, x, y)

Move the particle with index `idx` while applying periodic boundary conditions.
"""
@inline function move!(particles::RegularPolygons, idx::Integer, x::Real, y::Real)
    particles.centers[1, idx] += x
    particles.centers[2, idx] += y
    apply_parallelogram_boundary!(particles, idx)
end

@inline function apply_parallelogram_boundary!(particles::RegularPolygons, idx::Integer)
    preshift = particles.boxsize[1] / 2 - particles.centers[2, idx] * particles.boxshear[]
    particles.centers[2, idx] = mod(particles.centers[2, idx] + particles.boxsize[2] / 2,
        particles.boxsize[2]) - particles.boxsize[2] / 2
    postshift = particles.boxsize[1] / 2 - particles.centers[2, idx] * particles.boxshear[]
    particles.centers[1, idx] = mod(particles.centers[1, idx] + preshift,
        particles.boxsize[1]) - postshift
end

@inline function apply_parallelogram_boundary(
        particles::RegularPolygons, vec::Tuple{<:Real, <:Real})
    preshift = particles.boxsize[1] / 2 - vec[2] * particles.boxshear[]
    y = mod(vec[2] + particles.boxsize[2] / 2,
        particles.boxsize[2]) - particles.boxsize[2] / 2
    postshift = particles.boxsize[1] / 2 - y * particles.boxshear[]
    return (mod(vec[1] + preshift, particles.boxsize[1]) - postshift, y)
end
