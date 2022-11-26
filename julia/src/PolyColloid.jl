module PolyColloid

export Simulation, ForcefulCompressor, MoveSizeTuner, TrajectoryRecorder,
    run!, build_configuration!, crystallize!, particle_count, particle_area, boxarea

using RecipesBase
using StaticArrays
using Random
using Statistics: mean

include("colloid.jl")
include("overlap.jl")
include("celloverlap.jl")
include("constraint.jl")
include("record.jl")
include("update.jl")
include("simulate.jl")
include("operations.jl")
include("visualize.jl")

end
