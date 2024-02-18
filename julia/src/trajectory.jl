struct RegularPolygonsSnapshot
    sidenum::Integer
    radius::Real
    boxshear::Real
    boxsize::Tuple{<:Real, <:Real}
    centers::AbstractMatrix
    angles::AbstractVector
    time::Integer
end

struct RegularPolygonsTrajectory
    sidenum::Integer
    radius::Real
    boxshears::AbstractVector
    boxsizes::AbstractVector
    centers::AbstractVector
    angles::AbstractVector
    times::AbstractVector
end

struct TrajectoryReader
    filepath::String
end

RegularPolygonsSnapshot(particles::RegularPolygons) = CUDA.@allowscalar(
    RegularPolygonsSnapshot(particles.sidenum, particles.radius, particles.boxshear[],
        Tuple(particles.boxsize), Array(particles.centers), Array(particles.angles), 0))

RegularPolygons(snapshot::RegularPolygonsSnapshot; gpu=false) =
    RegularPolygons{eltype(snapshot.centers)}(
        snapshot.sidenum, snapshot.radius, snapshot.boxsize,
        snapshot.centers, snapshot.angles; gpu=gpu, boxshear=snapshot.boxshear)

function RegularPolygonsSnapshot(filepath::String, frame::Integer)
    jldopen(filepath) do f
        fstr = "frame$frame/"
        return RegularPolygonsSnapshot(f["sidenum"], f["radius"], f[fstr*"boxshear"],
            Tuple(f[fstr*"boxsize"]), f[fstr*"centers"], f[fstr*"angles"], f[fstr*"time"])
    end
end

function RegularPolygonsTrajectory(filepath::String)
    jldopen(filepath) do f
        sidenum, radius = f["sidenum"], f["radius"]
    
        boxsizes = Vector{typeof(f["frame1/boxsize"])}(undef, 0)
        boxshears = Vector{typeof(f["frame1/boxshear"])}(undef, 0)
        times = Vector{typeof(f["frame1/time"])}(undef, 0)

        numtype = eltype(f["frame1/centers"])
        centers = Vector{Matrix{numtype}}(undef, 0)
        angles = Vector{Vector{numtype}}(undef, 0)

        try
            frame = 1
            while true
                push!(boxshears, f["frame$frame/boxshear"])
                push!(boxsizes, f["frame$frame/boxsize"])
                push!(centers, f["frame$frame/centers"])
                push!(angles, f["frame$frame/angles"])
                push!(times, f["frame$frame/time"])
                frame += 1
            end
        catch e
            if !isa(e, KeyError)
                throw(e)
            end
        end
        return RegularPolygonsTrajectory(
            sidenum, radius, boxshears, boxsizes, centers, angles, times)
    end
end

function RegularPolygonsTrajectory(filepath::String, frames::AbstractUnitRange)
    jldopen(filepath) do f
        sidenum, radius = f["sidenum"], f["radius"]
    
        boxsizes = Vector{typeof(f["frame1/boxsize"])}(undef, 0)
        boxshears = Vector{typeof(f["frame1/boxshear"])}(undef, 0)
        times = Vector{typeof(f["frame1/time"])}(undef, 0)

        numtype = eltype(f["frame1/centers"])
        centers = Vector{Matrix{numtype}}(undef, 0)
        angles = Vector{Vector{numtype}}(undef, 0)

        for frame in frames
            push!(boxshears, f["frame$frame/boxshear"])
            push!(boxsizes, f["frame$frame/boxsize"])
            push!(centers, f["frame$frame/centers"])
            push!(angles, f["frame$frame/angles"])
            push!(times, f["frame$frame/time"])
        end
        return RegularPolygonsTrajectory(
            sidenum, radius, boxshears, boxsizes, centers, angles, times)
    end
end

Base.getindex(trajectory::RegularPolygonsTrajectory, frame::Int) = RegularPolygonsSnapshot(
    trajectory.sidenum, trajectory.radius, trajectory.boxshears[frame],
    ((trajectory.boxsizes[frame])[1], (trajectory.boxsizes[frame])[2]),
    trajectory.centers[frame], trajectory.angles[frame], trajectory.times[frame])

Base.getindex(trajectory::RegularPolygonsTrajectory, frames::AbstractUnitRange) =
    RegularPolygonsTrajectory(
        trajectory.sidenum, trajectory.radius, trajectory.boxshears[frames],
        trajectory.boxsizes[frames], trajectory.centers[frames],
        trajectory.angles[frames], trajectory.times[frames])

function Base.getindex(reader::TrajectoryReader, frame::Int)
    jldopen(reader.filepath) do f
        boxsize = f["frame$frame/boxsize"]
        return RegularPolygonsSnapshot(f["sidenum"], f["radius"], f["boxshear"],
            (boxsize[1], boxsize[2]), f["frame$frame/centers"], f["frame$frame/angles"],
            f["frame$frame/time"])
    end
end

function Base.getindex(reader::TrajectoryReader, frames::AbstractUnitRange)
    jldopen(reader.filepath) do f
        sidenum, radius = f["sidenum"], f["radius"]
    
        boxsizes = Vector{typeof(f["frame1/boxsize"])}(undef, 0)
        boxshears = Vector{typeof(f["frame1/boxshear"])}(undef, 0)
        times = Vector{typeof(f["frame1/time"])}(undef, 0)

        numtype = eltype(f["frame1/centers"])
        centers = Vector{Matrix{numtype}}(undef, 0)
        angles = Vector{Vector{numtype}}(undef, 0)

        for frame in frames
            push!(boxshears, f["frame$frame/boxshear"])
            push!(boxsizes, f["frame$frame/boxsize"])
            push!(centers, f["frame$frame/centers"])
            push!(angles, f["frame$frame/angles"])
            push!(times, f["frame$frame/time"])
            frame += 1
        end
        return RegularPolygonsTrajectory(
            sidenum, radius, boxshears, boxsizes, centers, angles, times)
    end
end

@inline Base.length(trajectory::RegularPolygonsTrajectory) = length(trajectory.times)

@inline particlecount(snapshot::RegularPolygonsSnapshot) = size(snapshot.centers, 2)

@inline particlearea(snapshot::RegularPolygonsSnapshot) = (
    0.5 * snapshot.sidenum * snapshot.radius^2 * sin(2π / snapshot.sidenum))
