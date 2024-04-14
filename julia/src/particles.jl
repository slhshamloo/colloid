abstract type ParticleCollection end

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

function RegularPolygons{T}(sidenum::Integer, radius::Real, boxsize::Tuple{<:Real, <:Real},
        centers::AbstractMatrix, angles::AbstractArray;
        boxshear::Real = 0.0, gpu::Bool = false) where {T<:Real}
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

@inline particlecount(particles::RegularPolygons) = size(particles.centers, 2)

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
