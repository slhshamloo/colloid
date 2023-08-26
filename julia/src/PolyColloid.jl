module PolyColloid

export Simulation, ForcefulCompressor, MoveSizeTuner, TrajectoryRecorder,
    run!, build_configuration!, crystallize!, particle_count, particle_area, boxarea

using CUDA
using StaticArrays
using Statistics
using Random
using JLD2
using RecipesBase
using Colors: Colorant
import Adapt

const numthreads=(16, 16)

include("colloid.jl")
include("overlap.jl")
include("cellseq.jl")
include("cellgpu.jl")
include("order.jl")
include("trajectory.jl")
include("constraint.jl")
include("tasks.jl")
include("simulate.jl")
include("simseq.jl")
include("simgpu.jl")
include("record.jl")
include("update.jl")
include("visualize.jl")

end
