abstract type AbstractRecorder end

struct ColloidSnapshot
    centers::AbstractMatrix
    angles::AbstractVector
    boxsize::Tuple{<:Real, <:Real}
end

mutable struct TrajectoryRecorder <: AbstractRecorder
    filepath::Union{String, Nothing}
    filecounter::Integer
    savetomem::Bool
    safe::Bool
    snapshots::Vector{ColloidSnapshot}
    times::Vector{Int}
    cond::Function

    function TrajectoryRecorder(cond::Function;
            filepath::Union{String, Nothing} = nothing,
            savetomem::Bool = false, safe::Bool = false)
        new(filepath, 0, savetomem, safe, ColloidSnapshot[], Int[], cond)
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
