struct Colloid{A<:AbstractArray, T<:Real}
    sidenum::Integer
    radius::T
    bisector::T # for speeding up calculations
    boxsize::MVector

    centers::A
    angles::A

    # for speeding up monte carlo step rejection
    _temp_centers::A
    _temp_angles::A

    function Colloid{A, T}(particle_count::Integer, sidenum::Integer, radius::Real,
                           boxsize::Tuple{<:Real, <:Real}) where {A<:AbstractArray, T<:Real}
        boxsize = MVector{2, T}(boxsize)
        bisector = radius * cos(π / sidenum)

        angles = A{T, 1}(undef, particle_count)
        centers = A{T, 2}(undef, 2, particle_count)

        temp_angles = copy(angles)
        temp_centers = copy(centers)

        new{A, T}(sidenum, radius, bisector, boxsize, centers, angles,
                  temp_centers, temp_angles)
    end
end

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

function crystallize!(colloid::Colloid)
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

    colloid.centers .= centers[:, 1:particle_count(colloid)]
    colloid._temp_centers .= colloid.centers
    colloid.angles .= 0
    colloid._temp_angles .= 0
end

@inline function apply_periodic_boundary!(colloid::Colloid)
    colloid.centers .-= colloid.centers ÷ (colloid.boxsize / 2) * colloid.boxsize
end

@inline function move!(colloid::Colloid, idx::Integer, x::Real, y::Real)
    colloid.centers[1, idx] += x
    colloid.centers[2, idx] += y
end

@inline function apply_periodic_boundary!(colloid::Colloid, idx::Integer)
    colloid.centers[1, idx] -= (
        colloid.centers[1, idx] ÷ (colloid.boxsize[1] / 2) * colloid.boxsize[1])
    colloid.centers[2, idx] -= (
        colloid.centers[2, idx] ÷ (colloid.boxsize[2] / 2) * colloid.boxsize[2])
end

@inline function rotate!(colloid::Colloid, idx::Integer, angle::Real)
    colloid.angles[idx] += angle
end
