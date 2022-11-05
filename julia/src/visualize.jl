@recipe function p(colloid::Colloid)
    boxsize = Array(colloid.boxsize)
    vertices = Array(colloid.vertices)
    for particle in 1:size(colloid.centers, 2)
        @series begin
            seriestype := :shape
            fillcolor --> :mediumblue
            linecolor --> :red
            xlim --> (-boxsize[1] / 2, boxsize[1] / 2)
            ylim --> (-boxsize[2] / 2, boxsize[2] / 2)
            vertices[1, :, particle], vertices[2, :, particle]
        end
    end
end
