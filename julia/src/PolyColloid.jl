module PolyColloid

export Colloid, simulate!, batchsim!, compress!, simple_compress!, crystal_initialize!,
    add_random_particle!, add_random_particles!

using RecipesBase
using StaticArrays
using Statistics: mean

include("polygon.jl")
include("overlap.jl")
include("colloid.jl")
include("simulate.jl")
include("visualize.jl")

end
