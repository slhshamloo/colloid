abstract type AbstractConstraint end

struct DiskWall{T<:Real} <: AbstractConstraint
    center::Tuple{T, T}
    radius::T
    update!::Function
end

struct RawConstraints{V<:AbstractVector, M<:AbstractMatrix}
    typeids::V
    data::M
end

Adapt.@adapt_structure RawConstraints

const max_data_per_constraint = 3
function build_raw_constraints(constraints::AbstractVector{<:AbstractConstraint},
                               type::DataType = Float32)
    typeids = Vector{Int}(undef, length(constraints))
    data = Matrix{type}(undef, max_data_per_constraint, length(constraints))
    for (i, constraint) in enumerate(constraints)
        if isa(constraint, DiskWall)
            typeids[i] = 0
            data[1, i] = constraint.center[1]
            data[2, i] = constraint.center[2]
            data[3, i] = constraint.radius
        end
    end
    return RawConstraints(CuArray(typeids), CuArray(data))
end

function is_violated(particles::RegularPolygons, disk::DiskWall, index::Integer)
    is_overlapping_with_disk(particles, index, disk.center, disk.radius) 
end

function fullcheck(particles::RegularPolygons, index::Integer,
        constraints::RawConstraints, cidx::Integer)
    if constraints.typeids[cidx] == 0
        return is_overlapping_with_disk(particles, index, (constraints.data[1, cidx],
            constraints.data[2, cidx]), constraints.data[3, cidx])
    else
        return false
    end
end

function fastcheck(particles::RegularPolygons, index::Integer,
        constraints::RawConstraints, cidx::Integer)
    if constraints.typeids[cidx] == 0
        return _overlap_range_disk(particles, index, (constraints.data[1, cidx],
            constraints.data[2, cidx]), constraints.data[3, cidx])
    else
        return false, false, (0.0f0, 0.0f0), 0.0f0
    end
end

function slowcheck(particles::RegularPolygons, index::Integer, dist::Tuple{<:Real, <:Real},
        distnorm::Real, constraints::RawConstraints, cidx::Integer)
    if constraints.typeids[cidx] == 0
        _is_disk_over_side(particles, index, dist, distnorm, constraints.data[3, cidx])
    else
        return false
    end
end

function has_violation(particles::RegularPolygons,
        constraints::AbstractVector{<:AbstractConstraint})
    for constraint in constraints
        if any(i -> is_violated(particles, constraint, i), 1:particlecount(particles))
            return true
        end
    end
    return false
end

function count_violations(particles::RegularPolygons,
        constraints::AbstractVector{<:AbstractConstraint})
    violations = 0
    for constraint in constraints
        violations += count(i -> is_violated(particles, constraint, i),
                            1:particlecount(particles))
    end
    return violations
end

function count_violations_gpu(particles::RegularPolygons,
        constraints::AbstractVector{<:AbstractConstraint})
    numblocks = particlecount(particles) ÷ numthreads + 1
    raw_constraints = build_raw_constraints(constraints, eltype(particles.centers))
    violation_counts = CuArray(zeros(Int32, numblocks))
    @cuda(threads=numthreads, blocks=numblocks, shmem = numthreads * sizeof(Int32),
          count_violations_parallel(particles, raw_constraints, violation_counts))
    return sum(violation_counts)
end

function count_violations_parallel(particles::RegularPolygons, constraints::RawConstraints,
                                   violation_counts::CuDeviceArray)
    thread = threadIdx().x
    tid = thread + (blockIdx().x - 1) * blockDim().x
    blockviolations = CuDynamicSharedArray(Int32, blockDim().x)
    blockviolations[thread] = 0

    if tid <= particlecount(particles)
        for cidx in 1:length(constraints.typeids)
            if fullcheck(particles, tid, constraints, cidx)
                blockviolations[thread] += 1
            end
        end
    end
    CUDA.sync_threads()

    i = blockDim().x ÷ 2
    while i != 0
        if thread <= i
            blockviolations[thread] += blockviolations[thread + i]
        end
        CUDA.sync_threads()
        i ÷= 2
    end
    if thread == 1
        violation_counts[blockIdx().x] = blockviolations[1]
    end
    return nothing
end
