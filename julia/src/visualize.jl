@recipe function p(colloid::Colloid)
    vertices = _build_vertices(colloid.sidenum, colloid.radius,
                               Array(colloid.centers), Array(colloid.angles))
    for particle in 1:particle_count(colloid)
        @series begin
            seriestype := :shape

            fillcolor --> :mediumblue
            linecolor --> :red
            ratio --> :equal
            legend --> false

            xlim --> (-colloid.boxsize[1] / 2, colloid.boxsize[1] / 2)
            ylim --> (-colloid.boxsize[2] / 2, colloid.boxsize[2] / 2)

            vertices[1, :, particle], vertices[2, :, particle]
        end
    end
end

@recipe function p(snapshot::ColloidSnapshot, sidenum::Integer, radius::Real)
    vertices = _build_vertices(sidenum, radius, snapshot.centers, snapshot.angles)
    for particle in 1:particle_count(colloid)
        @series begin
            seriestype := :shape

            fillcolor --> :mediumblue
            linecolor --> :red
            ratio --> :equal
            legend --> false

            xlim --> (-snapshot.boxsize[1] / 2, snapshot.boxsize[1] / 2)
            ylim --> (-snapshot.boxsize[2] / 2, snapshot.boxsize[2] / 2)

            vertices[1, :, particle], vertices[2, :, particle]
        end
    end
end
