module PolyColoid

export Coloid, RegPoly, RegEvenPoly, is_overlapping, mcsimulate!, mcsimulate_periodic!,
    crystal_initialize!, add_random_particle!, add_random_particles!

using StaticArrays

include("polygon.jl")
include("coloid.jl")

end
