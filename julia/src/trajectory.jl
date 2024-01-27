struct RegularPolygonsSnapshot
    sidenum::Integer
    radius::Real
    boxsize::Tuple{<:Real, <:Real}
    centers::AbstractMatrix
    angles::AbstractVector
    time::Integer
end

struct RegularPolygonsTrajectory
    sidenum::Integer
    radius::Real
    boxsizes::AbstractVector
    centers::AbstractVector
    angles::AbstractVector
    times::AbstractVector
end

struct TrajectoryReader
    filepath::String
end

RegularPolygonsSnapshot(particles::RegularPolygons) = RegularPolygonsSnapshot(
    particles.sidenum, particles.radius, Tuple(particles.boxsize),
    Array(particles.centers), Array(particles.angles), 0) 

RegularPolygons(snapshot::RegularPolygonsSnapshot; gpu=false) =
    RegularPolygons{eltype(snapshot.centers)}(
        snapshot.sidenum, snapshot.radius, snapshot.boxsize,
        snapshot.centers, snapshot.angles; gpu=gpu)

function RegularPolygonsTrajectory(filepath)
    jldopen(filepath) do f
        sidenum, radius = f["sidenum"], f["radius"]
    
        boxsizes = Vector{typeof(f["frame1/boxsize"])}(undef, 0)
        times = Vector{typeof(f["frame1/time"])}(undef, 0)

        numtype = eltype(f["frame1/centers"])
        centers = Vector{Matrix{numtype}}(undef, 0)
        angles = Vector{Vector{numtype}}(undef, 0)

        try
            frame = 1
            while true
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
        return RegularPolygonsTrajectory(sidenum, radius, boxsizes, centers, angles, times)
    end
end

Base.getindex(trajectory::RegularPolygonsTrajectory, frame::Int) = RegularPolygonsSnapshot(
    trajectory.sidenum, trajectory.radius,
    ((trajectory.boxsizes[frame])[1], (trajectory.boxsizes[frame])[2]),
    trajectory.centers[frame], trajectory.angles[frame], trajectory.times[frame])

Base.getindex(trajectory::RegularPolygonsTrajectory, frames::AbstractUnitRange) =
    RegularPolygonsTrajectory(
        trajectory.sidenum, trajectory.radius, trajectory.boxsizes[frames],
        trajectory.centers[frames], trajectory.angles[frames], trajectory.times[frames])

function Base.getindex(reader::TrajectoryReader, frame::Int)
    jldopen(reader.filepath) do f
        boxsize = f["frame$frame/boxsize"]
        return RegularPolygonsSnapshot(f["sidenum"], f["radius"], (boxsize[1], boxsize[2]),
            f["frame$frame/centers"], f["frame$frame/angles"], f["frame$frame/time"])
    end
end

function Base.getindex(reader::TrajectoryReader, frames::AbstractUnitRange)
    jldopen(reader.filepath) do f
        sidenum, radius = f["sidenum"], f["radius"]
    
        boxsizes = Vector{typeof(f["frame1/boxsize"])}(undef, 0)
        times = Vector{typeof(f["frame1/time"])}(undef, 0)

        numtype = eltype(f["frame1/centers"])
        centers = Vector{Matrix{numtype}}(undef, 0)
        angles = Vector{Vector{numtype}}(undef, 0)

        for frame in frames
            push!(boxsizes, f["frame$frame/boxsize"])
            push!(centers, f["frame$frame/centers"])
            push!(angles, f["frame$frame/angles"])
            push!(times, f["frame$frame/time"])
            frame += 1
        end
        return RegularPolygonsTrajectory(sidenum, radius, boxsizes, centers, angles, times)
    end
end

@inline Base.length(trajectory::RegularPolygonsTrajectory) = length(trajectory.times)

@inline particlecount(snapshot::RegularPolygonsSnapshot) = size(snapshot.centers, 2)

@inline particlearea(snapshot::RegularPolygonsSnapshot) = (
    0.5 * snapshot.sidenum * snapshot.radius^2 * sin(2π / snapshot.sidenum))
