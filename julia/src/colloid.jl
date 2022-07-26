struct Colloid{F<:AbstractFloat}
    particle_sidenum::Integer
    particle_radius::Real
    boxsize::AbstractVector

    particles::AbstractVector{<:AbstractPolygon}
    # for speeding up the reverting of monte carlo steps
    _temp_vertices::AbstractMatrix
    _temp_normals::AbstractMatrix
    _temp_center::AbstractVector

    function Colloid{F}(particle_count::Integer, particle_sidenum::Integer,
            particle_radius::Real, boxsize::Tuple{<:Real, <:Real}
            ) where {F<:AbstractFloat}
        if particle_sidenum % 2 == 0
            P = RegEvenPoly
        else
            P = RegPoly
        end
        boxsize = MVector{2, F}(boxsize)
        particles = [P{F}(particle_sidenum, particle_radius, rand(), Tuple(boxsize .* ratio))
            for ratio in eachcol(rand(F, 2, particle_count))]
        new(particle_sidenum, particle_radius, boxsize, particles,
            similar(particles[1].vertices), similar(particles[1].normals),
            similar(particles[1].center))
    end
end

function Colloid(particle_count::Integer, particle_sidenum::Integer,
        particle_radius::Real, boxsize::Tuple{<:Real, <:Real})
    Colloid{Float64}(particle_count, particle_sidenum, particle_radius, boxsize)
end

function crystal_initialize!(colloid::Colloid, gridwidth::Integer,
        dist::Tuple{<:Real, <:Real}, offset::Tuple{<:Real, <:Real})
    for i in 1:length(colloid.particles)
        particle = colloid.particles[i]
        new_center = (offset[1] + dist[1] * ((i-1) ÷ gridwidth),
            offset[2] + dist[2] * ((i-1) % gridwidth))
        warp = new_center .- particle.center
        particle.center .= new_center
        particle.vertices .+= warp
    end
end

nematic_order(colloid::Colloid) = mean(nematic_order, colloid.particles)

function nematic_order(poly::AbstractPolygon)
    return (3 * max(poly.normals[1]^2, poly.normals[2]^2) - 1) / 2
end

function body_order(colloid::Colloid, index::Integer)
    particle = colloid.particles[index]
    return (particle.normals[1] + 1im * particle.normals[2]) * mean(
        p -> p.normals[1] + 1im * p.normals[2], colloid.particles)
end

function add_random_particle!(colloid::Colloid)
    new_particle = eltype(colloid.particles)(colloid.particle_sidenum, colloid.particle_radius,
        2π / colloid.particle_sidenum * rand(),
        (colloid.boxsize[1] * rand(), colloid.boxsize[2] * rand()))
    for particle in colloid.particles
        if is_overlapping(particle, new_particle)
            return
        end
    end
    push!(colloid.patricles, new_particle)
end

function add_random_particles!(colloid::Colloid, count::Integer)
    rnd = rand(3, count)
    for i in 1:count
        new_particle = eltype(colloid.particles)(colloid.particle_sidenum,
            colloid.particle_radius, 2π / colloid.particle_sidenum * rnd[1, i],
            (colloid.boxsize[1] * rnd[2, i], colloid.boxsize[2] * rnd[3, i]))
        flag = true
        for particle in colloid.particles
            if is_overlapping(particle, new_particle)
                flag = false
                break
            end
        end
        if flag
            push!(colloid.particles, new_particle)
        end
    end
end

periodic_boundary_shift(boxsize) = (
    x -> (-(x[1] ÷ (boxsize[1]/2)) * boxsize[1],
        -(x[2] ÷ (boxsize[2]/2)) * boxsize[2]))
