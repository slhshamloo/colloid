@recipe function p(snapshot::RegularPolygonsSnapshot)
    vertices = _build_vertices(snapshot.sidenum, snapshot.radius,
                               snapshot.centers, snapshot.angles)
    shearside = snapshot.boxsize[1] + abs(snapshot.boxsize[2] * snapshot.boxshear[])
    for particle in 1:particlecount(snapshot)
        @series begin
            seriestype := :shape

            fillcolor --> :mediumblue
            linecolor --> :red
            linewidth --> (80 * snapshot.radius * cos(π / snapshot.sidenum)
                           / maximum(snapshot.boxsize))
            ratio --> :equal
            legend --> false

            xlim --> (-shearside / 2, shearside / 2)
            ylim --> (-snapshot.boxsize[2] / 2, snapshot.boxsize[2] / 2)

            vertices[1, :, particle], vertices[2, :, particle]
        end
    end
end

@recipe function p(snapshot::RegularPolygonsSnapshot, colors::Vector{<:Colorant})
    vertices = _build_vertices(snapshot.sidenum, snapshot.radius,
                               snapshot.centers, snapshot.angles)
    shearside = snapshot.boxsize[1] + abs(snapshot.boxsize[2] * snapshot.boxshear[])
    for particle in 1:particlecount(snapshot)
        @series begin
            seriestype := :shape

            fillcolor --> colors[particle]
            linecolor --> :transparent
            linewidth --> 0
            ratio --> :equal
            legend --> false

            xlim --> (-shearside / 2, shearside / 2)
            ylim --> (-snapshot.boxsize[2] / 2, snapshot.boxsize[2] / 2)

            vertices[1, :, particle], vertices[2, :, particle]
        end
    end
end

@recipe p(snapshot::RegularPolygonsSnapshot, colors::Vector{<:AbstractFloat},
          scheme::ColorScheme = vikO) = (snapshot, [get(scheme, color) for color in colors])

@recipe p(::Type{<:RegularPolygons}, particles::RegularPolygons) =
    RegularPolygonsSnapshot(particles)
