struct Colloid{T<:Real, V<:AbstractVector, M<:AbstractMatrix, VB<:AbstractVector}
    sidenum::Int
    radius::T
    bisector::T
    boxsize::VB

    centers::M
    angles::V
end

function Colloid{T}(particle_count::Integer, sidenum::Integer, radius::Real,
        boxsize::Tuple{<:Real, <:Real}; gpu=false) where {T<:Real}
    bisector = radius * cos(π / sidenum)

    if gpu
        angles = CuVector{T}(undef, particle_count)
        centers = CuMatrix{T}(undef, 2, particle_count)
        boxsize = CuVector{T}([boxsize[1], boxsize[2]])
    else
        angles = Vector{T}(undef, particle_count)
        centers = Matrix{T}(undef, 2, particle_count)
        boxsize = MVector{2, T}(boxsize)
    end

    Colloid{T, typeof(angles), typeof(centers), typeof(boxsize)}(
        sidenum, radius, bisector, boxsize, centers, angles)
end

Adapt.@adapt_structure Colloid

@inline particle_count(colloid::Colloid) = size(colloid.centers, 2)

@inline particle_area(colloid::Colloid) = (
    0.5 * colloid.sidenum * colloid.radius^2 * sin(2π / colloid.sidenum))

@inline boxarea(colloid::Colloid) = colloid.boxsize[1] * colloid.boxsize[2]

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

function crystallize!(colloid::Colloid,
        constraint::Function = (colloid, centers) -> trues(size(centers, 2)))
    particles_per_side = ceil(Int, √(particle_count(colloid)))
    shortside = minimum(colloid.boxsize)
    shortdim = argmin(colloid.boxsize)
    spacing = shortside / particles_per_side

    repvals = range((-shortside + spacing) / 2, (shortside - spacing) / 2,
                    particles_per_side)
    shortpos = repeat(repvals, particles_per_side)
    longpos = repeat(repvals, inner=particles_per_side)

    xs = shortdim == 1 ? shortpos : longpos
    ys = shortdim == 2 ? shortpos : longpos
    centers = permutedims(hcat(xs, ys))
    valid = constraint(colloid, centers)
    centers = centers[vcat(valid, valid)]
    centers = reshape(centers, 2, length(centers)÷2)
    centers = centers[:, 1:particle_count(colloid)]

    if isa(colloid.centers, CuArray)
        centers = CuArray(centers)
    end
    colloid.centers .= centers
    colloid.angles .= 0
end

@inline function move!(colloid::Colloid, idx::Integer, x::Real, y::Real)
    colloid.centers[1, idx] += x
    colloid.centers[2, idx] += y
    # apply periodic boundary conditions
    colloid.centers[1, idx] -= (
        colloid.centers[1, idx] ÷ (colloid.boxsize[1] / 2) * colloid.boxsize[1])
    colloid.centers[2, idx] -= (
        colloid.centers[2, idx] ÷ (colloid.boxsize[2] / 2) * colloid.boxsize[2])
end
