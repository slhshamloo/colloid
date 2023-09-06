@recipe function p(snapshot::ColloidSnapshot)
    vertices = _build_vertices(snapshot.sidenum, snapshot.radius,
                               snapshot.centers, snapshot.angles)
    for particle in 1:particle_count(snapshot)
        @series begin
            seriestype := :shape

            fillcolor --> :mediumblue
            linecolor --> :red
            linewidth --> (80 * snapshot.radius * cos(π / snapshot.sidenum)
                           / maximum(snapshot.boxsize))
            ratio --> :equal
            legend --> false

            xlim --> (-snapshot.boxsize[1] / 2, snapshot.boxsize[1] / 2)
            ylim --> (-snapshot.boxsize[2] / 2, snapshot.boxsize[2] / 2)

            vertices[1, :, particle], vertices[2, :, particle]
        end
    end
end

@recipe function p(snapshot::ColloidSnapshot, colors::Vector{<:Colorant})
    vertices = _build_vertices(snapshot.sidenum, snapshot.radius,
                               snapshot.centers, snapshot.angles)
    for particle in 1:particle_count(snapshot)
        @series begin
            seriestype := :shape

            fillcolor --> colors[particle]
            linecolor --> :transparent
            linewidth --> 0
            ratio --> :equal
            legend --> false

            xlim --> (-snapshot.boxsize[1] / 2, snapshot.boxsize[1] / 2)
            ylim --> (-snapshot.boxsize[2] / 2, snapshot.boxsize[2] / 2)

            vertices[1, :, particle], vertices[2, :, particle]
        end
    end
end

@recipe p(::Type{<:Colloid}, colloid::Colloid) = ColloidSnapshot(colloid)
