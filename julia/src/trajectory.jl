"""
    RegularPolygonsSnapshot

Structure for holding system information in one time step of the simulation.
"""
struct RegularPolygonsSnapshot
    sidenum::Integer
    radius::Real
    boxshear::Real
    boxsize::Tuple{<:Real, <:Real}
    centers::AbstractMatrix
    angles::AbstractVector
    time::Integer
end


"""
    RegularPolygonsSnapshot

Structure for holding system information in multiple time steps of the simulation.
"""
struct RegularPolygonsTrajectory
    sidenum::Integer
    radius::Real
    boxshears::AbstractVector
    boxsizes::AbstractVector
    centers::AbstractVector
    angles::AbstractVector
    times::AbstractVector
end

"""
    RegularPolygonsSnapshot(particles::RegularPolygons)

Convert particle collection into snapshot.
"""
RegularPolygonsSnapshot(particles::RegularPolygons) = CUDA.@allowscalar(
    RegularPolygonsSnapshot(particles.sidenum, particles.radius, particles.boxshear[],
        Tuple(particles.boxsize), Array(particles.centers), Array(particles.angles), 0))

"""
    RegularPolygons(snapshot::RegularPolygonsSnapshot; gpu=false)

Convert snapshot into particle collection.
"""
RegularPolygons(snapshot::RegularPolygonsSnapshot; gpu::Bool = false) =
    RegularPolygons{eltype(snapshot.centers)}(
        snapshot.sidenum, snapshot.radius, snapshot.boxsize,
        snapshot.centers, snapshot.angles; gpu=gpu, boxshear=snapshot.boxshear)

"""
    RegularPolygonsSnapshot(filepath, frame)

Load `frame` from `filepath` into a snapshot.
"""
function RegularPolygonsSnapshot(filepath::String, frame::Integer)
    if !endswith(filepath, ".jld2")
        filepath *= ".jld2"
    end
    jldopen(filepath) do f
        fstr = "frame$frame/"
        return RegularPolygonsSnapshot(f["sidenum"], f["radius"], f[fstr*"boxshear"],
            Tuple(f[fstr*"boxsize"]), f[fstr*"centers"], f[fstr*"angles"], f[fstr*"time"])
    end
end

"""
    RegularPolygonsSnapshot(filepath)

Load every frame from `filepath` into a trajectory.
"""
function RegularPolygonsTrajectory(filepath::String)
    if !endswith(filepath, ".jld2")
        filepath *= ".jld2"
    end
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

"""
    RegularPolygonsSnapshot(filepath, frames)

Load `frames` from `filepath` into a trajectory.
"""
function RegularPolygonsTrajectory(filepath::String, frames::OrdinalRange)
    if !endswith(filepath, ".jld2")
        filepath *= ".jld2"
    end
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

"""
    framecount(filepath)

count the number of frames in trajectory saved in `filepath`.
"""
function framecount(filepath::String)
    if !endswith(filepath, ".jld2")
        filepath *= ".jld2"
    end
    frame = 1
    jldopen(filepath) do f
        try
            while true
                time = f["frame$frame/time"]
                frame += 1
            end
        catch e
            frame -= 1
            if !isa(e, KeyError)
                throw(e)
            end
        end
    end
    return frame
end

Base.getindex(trajectory::RegularPolygonsTrajectory, frame::Int) = RegularPolygonsSnapshot(
    trajectory.sidenum, trajectory.radius, trajectory.boxshears[frame],
    ((trajectory.boxsizes[frame])[1], (trajectory.boxsizes[frame])[2]),
    trajectory.centers[frame], trajectory.angles[frame], trajectory.times[frame])

Base.getindex(trajectory::RegularPolygonsTrajectory, frames::OrdinalRange) =
    RegularPolygonsTrajectory(
        trajectory.sidenum, trajectory.radius, trajectory.boxshears[frames],
        trajectory.boxsizes[frames], trajectory.centers[frames],
        trajectory.angles[frames], trajectory.times[frames])

@inline Base.length(trajectory::RegularPolygonsTrajectory) = length(trajectory.times)

@inline particlecount(snapshot::RegularPolygonsSnapshot) = size(snapshot.centers, 2)

@inline particlearea(snapshot::RegularPolygonsSnapshot) = (
    0.5 * snapshot.sidenum * snapshot.radius^2 * sin(2π / snapshot.sidenum))

"""
    calculate_local_order(trajectory, type, typeparams...; gpu=false, numtype=Float32, typekeywords...)

Calculate order parameters from `trajectory` structure.
"""
function calculate_local_order(trajectory::RegularPolygonsTrajectory,
        type::String, typeparams...; gpu::Bool = false, numtype::DataType = Float32,
        typekeywords...)
    if type == "nematic"
        ordertype = numtype
    elseif type == "solidliquid"
        ordertype = UInt32
    else
        ordertype = Complex{numtype}
    end
    orders = Vector{Vector{ordertype}}(undef, 0)
    for frame in 1:length(trajectory)
        particles = RegularPolygons(trajectory[frame], gpu=gpu)
        if type == "katic"
            push!(orders, katic_order(particles, typeparams...; numtype=numtype))
        elseif type == "solidliquid"
            push!(orders, solidliquid(particles, typeparams...; typekeywords...))
        end
    end
    return orders
end

"""
    calculate_local_order(filepath, frame, type, typeparams...; gpu=false, numtype=Float32, typekeywords...)

Calculate order parameters in `frames` from trajectory recorded in `filepath`.
"""
function calculate_local_order(filepath::String, frames::OrdinalRange, 
        type::String, typeparams...; gpu::Bool = false, numtype::DataType = Float32)
    if type == "nematic"
        ordertype = numtype
    elseif type == "solidliquid"
        ordertype = Int32
    else
        ordertype = Complex{numtype}
    end
    orders = Vector{Vector{ordertype}}(undef, 0)
    for frame in frames
        particles = RegularPolygons(RegularPolygonsSnapshot(filepath, frame), gpu=gpu)
        cell_list = gpu ? CuCellList(particles) : SeqCellList(particles)
        if type == "katic"
            push!(orders, katic_order(particles, cell_list, typeparams...; numtype=numtype))
        elseif type == "solidliquid"
            push!(orders, solidliquid(particles, cell_list, typeparams...))
        end
    end
    return orders
end
