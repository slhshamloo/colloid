struct Colloid{A<:AbstractArray, T<:Real}
    sidenum::Integer
    radius::T
    bisector::T # for speeding up calculations
    boxsize::MVector

    centers::A
    vertices::A
    normals::A

    # for speeding up monte carlo step rejection
    _temp_shifts::A
    _temp_centers::A
    _temp_vertices::A
    _temp_normals::A

    function Colloid{A, T}(particle_count::Integer, sidenum::Integer, radius::Real,
                           boxsize::Tuple{<:Real, <:Real}) where {A<:AbstractArray, T<:Real}
        boxsize = MVector{2, T}(boxsize)
        bisector = radius * cos(π / sidenum)
        normalnum = sidenum % 2 == 0 ? sidenum ÷ 2 : sidenum

        centers = Matrix{T}(undef, 2, particle_count)
        vertices, normals = _build_vertices_and_normals(
            T, sidenum, normalnum, radius, centers, zeros(particle_count))
        centers = _build_array(A, T, (2, particle_count), centers)
        vertices = _build_array(A, T, (2, sidenum, particle_count), vertices)
        normals = _build_array(A, T, (2, normalnum, particle_count), normals)

        temp_shifts = similar(centers)
        temp_centers = copy(centers)
        temp_vertices = copy(vertices)
        temp_normals = copy(normals)

        new{A, T}(sidenum, radius, bisector, boxsize, centers, vertices, normals,
                  temp_shifts, temp_centers, temp_vertices, temp_normals)
    end
end

@inline particle_count(colloid::Colloid) = size(colloid.centers, 2)

function _build_array(A::UnionAll, T::DataType, dims::Tuple{Vararg{<:Integer}}, content)
    if A <: StaticArray
        return A{Tuple{dims...}, T}(content)
    else
        return A{T, length(dims)}(content)
    end
end

function _build_vertices_and_normals(T::DataType, sidenum::Integer, normalnum::Integer,
        radius::Real, centers::AbstractMatrix, angles::AbstractVector)
    vertices = Array{T, 3}(undef, 2, sidenum, size(centers, 2))
    normals = Array{T, 3}(undef, 2, normalnum, size(centers, 2))
    for particle in eachindex(angles)
        θs = (k * π / sidenum + angles[particle] for k in 0:2:2sidenum-2)
        @. vertices[1, :, particle] = radius * cos(θs)
        @. vertices[2, :, particle] = radius * sin(θs)
        vertices[:, :, particle] .+= centers[:, particle]

        θs = (k * π / sidenum + angles[particle] for k in 1:2:2normalnum-1)
        @. normals[1, :, particle] = cos(θs)
        @. normals[2, :, particle] = sin(θs)
    end
    return vertices, normals
end

function build_configuration!(colloid::Colloid, centers::AbstractMatrix,
                              angles::AbstractVector)
    vertices, normals = _build_vertices_and_normals(
        eltype(colloid.centers), colloid.sidenum, size(colloid.normals, 2),
        colloid.radius, centers, angles)
    colloid.centers .= centers
    colloid._temp_centers .= centers
    colloid.vertices .= vertices
    colloid._temp_vertices .= vertices
    colloid.normals .= normals
    colloid._temp_normals .= normals
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
    centers = centers[:, 1:particle_count(colloid)]
    build_configuration!(colloid, centers, zeros(particle_count(colloid)))
end

@inline function apply_periodic_boundary!(colloid::Colloid)
    @. colloid._temp_shifts = -colloid.centers ÷ (colloid.boxsize / 2) * colloid.boxsize
    colloid.centers .+= colloid._temp_shifts
    colloid.vertices .+= reshape(colloid._temp_shifts, 2, 1, particle_count(colloid))
end

@inline function move!(colloid::Colloid, idx::Integer, x::Real, y::Real)
    colloid.centers[1, idx] += x
    colloid.centers[2, idx] += y
    colloid.vertices[:, :, idx] .+= (x, y)
end

@inline function apply_periodic_boundary!(colloid::Colloid, idx::Integer)
    shift = (-colloid.centers[1, idx] ÷ (colloid.boxsize[1] / 2) * colloid.boxsize[1],
        -colloid.centers[2, idx] ÷ (colloid.boxsize[2] / 2) * colloid.boxsize[2])
    colloid.centers[1, idx] += shift[1]
    colloid.centers[2, idx] += shift[2]
    colloid.vertices[:, :, idx] .+= shift
end

@inline function rotate!(colloid::Colloid, idx::Integer, angle::Real)
    colloid.vertices[:, :, idx] .-= colloid.centers[:, idx]
    for v in 1:colloid.sidenum
        temp = (colloid.vertices[1, v, idx] * cos(angle)
            - colloid.vertices[2, v, idx] * sin(angle))
        colloid.vertices[2, v, idx] = (colloid.vertices[1, v, idx] * sin(angle)
            + colloid.vertices[2, v, idx] * cos(angle))
        colloid.vertices[1, v, idx] = temp
    end
    colloid.vertices[:, :, idx] .+= colloid.centers[:, idx]

    for n in 1:size(colloid.normals, 2)
        temp = (colloid.normals[1, n, idx] * cos(angle)
            - colloid.normals[2, n, idx] * sin(angle))
        colloid.normals[2, n, idx] = (colloid.normals[1, n, idx] * sin(angle)
            + colloid.normals[2, n, idx] * cos(angle))
        colloid.normals[1, n, idx] = temp
    end
end
