function _get_nonperiodic_corners(poly::AbstractPolygon)
    θ₀ = π / poly.sidenum

    if abs(poly.normals[1]) > 1
        poly.normals[1] = round(poly.normals[1])
    end
    angle = acos(poly.normals[1]) * sign(poly.normals[2])
    θs = [k * θ₀ + angle for k in 1:2:2poly.sidenum-1]

    poly.center[1] .+ poly.radius * cos.(θs), poly.center[2] .+ poly.radius * sin.(θs)
end

@recipe function p(poly::AbstractPolygon)
    @series begin
        seriestype := :shape
        fillcolor --> :mediumblue
        linecolor --> :red
        _get_nonperiodic_corners(poly)
    end
end

@recipe function p(polys::AbstractVector{<:AbstractPolygon})
    for poly in polys
        @series begin
            seriestype := :shape
            fillcolor --> :mediumblue
            linecolor --> :red
            _get_nonperiodic_corners(poly)
        end
    end
end

@recipe p(colloid::colloid) = colloid.particles
