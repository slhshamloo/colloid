abstract type AbstractRecorder end

struct ColloidSnapshot
    centers::AbstractMatrix
    angles::AbstractVector
    boxsize::Tuple{<:Real, <:Real}
end

struct TrajectoryRecorder <: AbstractRecorder
    snapshots::Vector{ColloidSnapshot}
    times::Vector{Int}
    cond::Function

    function TrajectoryRecorder(cond::Function)
        new(ColloidSnapshot[], Int[], cond)
    end
end

struct LocalOrderRecorder{T <: Number} <: AbstractRecorder
    type::String
    typeparams::Tuple{Vararg{<:Real}}
    orders::Vector{Vector{T}}
    times::Vector{Int}
    cond::Function

    function LocalOrderRecorder(cond::Function, type::String, typeparams...;
                                numtype::DataType = Float32)
        if type != "nematic"
            numtype = Complex{numtype}
        end
        orders = Vector{Vector{numtype}}(undef, 0)
        new{numtype}(type, typeparams, orders, Int[], cond)
    end
end

struct GlobalOrderRecorder{T <: Number} <: AbstractRecorder
    type::String
    typeparams::Tuple{Vararg{<:Real}}
    orders::Vector{T}
    times::Vector{Int}
    cond::Function

    function GlobalOrderRecorder(cond::Function, type::String, typeparams...;
                                 numtype::DataType = Float32)
        if type != "nematic"
            numtype = Complex{numtype}
        end
        orders = Vector{numtype}(undef, 0)
        new{numtype}(type, typeparams, orders, Int[], cond)
    end
end

function get_snapshot(colloid::Colloid)
    return ColloidSnapshot(Matrix(colloid.centers), Vector(colloid.angles),
                           Tuple(colloid.boxsize))
end

function katic_order_term(colloid::Colloid, k::Integer, i::Integer, j::Integer)
    rij = (colloid.centers[1, i] - colloid.centers[1, j],
           colloid.centers[2, i] - colloid.centers[2, j])
    rij = (rij[1] - rij[1] ÷ (colloid.boxsize[1]/2) * colloid.boxsize[1],
           rij[2] - rij[2] ÷ (colloid.boxsize[2]/2) * colloid.boxsize[2])
    angle = (rij[2] < 0 ? -1 : 1) * acos(rij[1] / sqrt(rij[1]^2 + rij[2]^2))
    return exp(1im * k * angle)
end

function katic_order(colloid::Colloid, cell_list::CuCellList, k::Integer)
    blockthreads = (numthreads[1] * numthreads[2])
    maxcount = maximum(cell_list.counts)
    groupcount = 9 * maxcount
    groups_per_block = blockthreads ÷ groupcount
    numblocks = particle_count(colloid) ÷ groups_per_block + 1
    orders = CuVector{ComplexF32}(undef, particle_count(colloid))
    @cuda(threads=blockthreads, blocks=numblocks,
          shmem = 3 * groups_per_block * sizeof(Float32),
          katic_order_parallel(colloid, cell_list, orders, k, maxcount,
                               groupcount, groups_per_block))
    return Vector(orders)
end

function katic_order_parallel(colloid::Colloid, cell_list::CuCellList,
        orders::CuDeviceVector, k::Integer, maxcount::Integer,
        groupcount::Integer, groups_per_block::Integer)
    shared_memory = CuDynamicSharedArray(Float32, 3 * groups_per_block)
    group_orders_real = @view shared_memory[1:groups_per_block]
    group_orders_imag = @view shared_memory[groups_per_block+1:2groups_per_block]
    neighbor_counts = @view shared_memory[2groups_per_block+1:end]

    is_thread_active = threadIdx().x <= groups_per_block * groupcount
    if is_thread_active
        group, thread = divrem(threadIdx().x - 1, groupcount)
        group += 1
        particle = (blockIdx().x - 1) * groups_per_block + group
    
        if particle <= particle_count(colloid)
            i, j = get_cell_list_indices(colloid, cell_list, particle)
            if thread == 0
                group_orders_real[group] = 0.0f0
                group_orders_imag[group] = 0.0f0
                neighbor_counts[group] = count_neighbors(cell_list, i, j)
            end
        else
            is_thread_active = false
        end
    end
    CUDA.sync_threads()

    if is_thread_active && neighbor_counts[group] != 0
        katic_order_neighbor(colloid, cell_list, group_orders_real, group_orders_imag,
                             k, Int(neighbor_counts[group]), particle, i, j, maxcount,
                             group, thread)
    end
    CUDA.sync_threads()

    if is_thread_active && thread == 0
        orders[particle] = group_orders_real[group] + 1im * group_orders_imag[group]
    end
    return
end

function katic_order_neighbor(colloid::Colloid, cell_list::CuCellList,
        group_orders_real::SubArray, group_orders_imag::SubArray,
        k::Integer, neighbor_count::Integer, particle::Integer,
        i::Integer, j::Integer, maxcount::Integer, group::Integer, thread::Integer)
    relpos, kneighbor = divrem(thread, maxcount)
    kneighbor += 1
    jdelta, idelta = divrem(relpos, 3)
    ineighbor = mod(i + idelta - 2, size(cell_list.cells, 2)) + 1
    jneighbor = mod(j + jdelta - 2, size(cell_list.cells, 3)) + 1

    if kneighbor <= cell_list.counts[ineighbor, jneighbor]
        neighbor = cell_list.cells[kneighbor, ineighbor, jneighbor]
        if particle != neighbor
            katic = katic_order_term(colloid, k, particle, neighbor) / neighbor_count
            CUDA.@atomic group_orders_real[group] += real(katic)
            CUDA.@atomic group_orders_imag[group] += imag(katic)
        end
    end
    return
end

function count_neighbors(cell_list::CuCellList, i::Integer, j::Integer)
    return (
        cell_list.counts[mod(i - 2, size(cell_list.counts, 1)) + 1,
                         mod(j - 2, size(cell_list.counts, 2)) + 1]
        + cell_list.counts[mod(i - 2, size(cell_list.counts, 1)) + 1, j]
        + cell_list.counts[mod(i - 2, size(cell_list.counts, 1)) + 1,
                           mod(j, size(cell_list.counts, 1)) + 1]
        + cell_list.counts[i, mod(j - 2, size(cell_list.counts, 1)) + 1]
        + cell_list.counts[i, j]
        + cell_list.counts[i, mod(j, size(cell_list.counts, 1)) + 1]
        + cell_list.counts[mod(i, size(cell_list.counts, 1)) + 1,
                         mod(j - 2, size(cell_list.counts, 2)) + 1]
        + cell_list.counts[mod(i, size(cell_list.counts, 1)) + 1, j]
        + cell_list.counts[mod(i, size(cell_list.counts, 1)) + 1,
                           mod(j, size(cell_list.counts, 1)) + 1]
    )
end
