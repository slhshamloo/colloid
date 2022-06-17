module PolyColoid

export Coloid, mcsimulate!, mcsimulate_periodic!, crystal_initialize!,
    add_random_particle!, add_random_particles!

using RecipesBase
using StaticArrays
using Statistics: mean

include("polygon.jl")
include("overlap.jl")
include("potential.jl")
include("coloid.jl")
include("visualize.jl")

end
