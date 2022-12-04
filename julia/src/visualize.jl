@recipe function p(colloid::Colloid)
    vertices = _build_vertices(colloid.sidenum, colloid.radius,
                               colloid.centers, colloid.angles)
    for particle in 1:particle_count(colloid)
        @series begin
            seriestype := :shape
            fillcolor --> :mediumblue
            linecolor --> :red
            xlim --> (-colloid.boxsize[1] / 2, colloid.boxsize[1] / 2)
            ylim --> (-colloid.boxsize[2] / 2, colloid.boxsize[2] / 2)
            vertices[1, :, particle], vertices[2, :, particle]
        end
    end
end
