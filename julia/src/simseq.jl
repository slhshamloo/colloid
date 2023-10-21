function apply_step_cpu!(sim::HPMCSimulation)
    randchoices = rand(Bool, count(sim.particles))
    randnums = rand(sim.numtype,
        2 + Int(!isnothing(sim.potential) || !isnothing(sim.pairpotential)),
        count(sim.particles))
    if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
        if length(sim.particle_potentials) == 0
            sim.particle_potentials = zero(sim.particles.angles)
        end
        calculate_potentials!(sim.particles, sim.cell_list, sim.potential,
                              sim.pairpotential, sim.particle_potentials)
    end
    
    iter = (rand(Bool) ?
        range(1, count(sim.particles))
        : range(count(sim.particles), 1, step=-1)
    )
    for idx in iter
        if randchoices[idx]
            apply_translation!(sim, randnums, idx)
        else
            apply_rotation!(sim, randnums, idx)
        end
    end
    return true
end

function apply_translation!(sim::HPMCSimulation, randnums::Matrix{<:Real}, idx::Int)
    r = sim.move_radius * randnums[1, idx]
    θ = 2π * randnums[2, idx]
    x, y = r * cos(θ), r * sin(θ)

    i, j = get_cell_list_indices(sim.particles, sim.cell_list, idx)
    deleteat!(sim.cell_list.cells[i, j], findfirst(==(idx), sim.cell_list.cells[i, j]))
    move!(sim.particles, idx, x, y)
    i, j = get_cell_list_indices(sim.particles, sim.cell_list, idx)
    push!(sim.cell_list.cells[i, j], idx)

    if has_violation(sim, randnums, idx, i, j)
        move!(sim.particles, idx, -x, -y)
        pop!(sim.cell_list.cells[i, j])
        i, j = get_cell_list_indices(sim.particles, sim.cell_list, idx)
        push!(sim.cell_list.cells[i, j], idx)
        sim.rejected_translations += 1
    else
        sim.accepted_translations += 1
    end
end

function apply_rotation!(sim::HPMCSimulation,randnums::Matrix{<:Real}, idx::Int)
    angle_change = sim.rotation_span * (randnums[2, idx] - 0.5)
    sim.particles.angles[idx] += angle_change
    i, j = get_cell_list_indices(sim.particles, sim.cell_list, idx)
    if has_violation(sim, randnums, idx, i, j)
        sim.particles.angles[idx] -= angle_change
        sim.rejected_rotations += 1
    else
        sim.accepted_rotations += 1
    end
end

@inline function has_violation(sim::HPMCSimulation, randnums::Matrix{<:Real},
                               idx::Integer, i::Integer, j::Integer)
    if (violates_constraints(sim, idx) || (isnothing(sim.pairpotential)
            && has_overlap(sim.particles, sim.cell_list, idx, i, j)))
        return true
    elseif !isnothing(sim.potential) || !isnothing(sim.pairpotential)
        potsum = zero(eltype(sim.particle_potentials))
        if !isnothing(sim.pairpotential)
            potsum += pairpotential_sum(
                sim.particles, sim.cell_list, sim.pairpotential, idx, i, j)
        end
        if !isnothing(sim.potential)
            potsum += sim.potential(sim.particles, idx)
        end
        potchange = potsum - sim.particle_potentials[idx]
        return randnums[3, idx] > exp(-sim.beta * potchange)
    else
        return false
    end
end

@inline function violates_constraints(sim::HPMCSimulation, idx::Integer)
    for constraint in sim.constraints
        if is_violated(sim.particles, constraint, idx)
            return true
        end
    end
    return false
end
