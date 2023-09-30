module PolyColloid

export ColloidSim, ForcefulCompressor, MoveSizeTuner, TrajectoryRecorder,
    run!, build_configuration!, crystallize!, pcount, parea, boxarea

using CUDA
using StaticArrays
using Statistics
using Random
using JLD2
using RecipesBase
using Colors: Colorant
using ColorSchemes: ColorScheme, vikO, get
import Adapt
import Threads

const numthreads=256

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
