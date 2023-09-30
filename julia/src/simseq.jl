function apply_step_cpu!(sim::ColloidSim)
    randchoices = rand(Bool, pcount(sim.colloid))
    randnums = rand(sim.numtype,
        2 + !isnothing(sim.potential) || !isnothing(sim.pairpotential),
        pcount(sim.colloid))
    if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
        calculate_potentials!(sim.colloid, sim.cell_list, sim.potential,
                              sim.pairpotential, sim.particle_potentials)
    end
    
    iter = (rand(Bool) ?
        range(1, pcount(sim.colloid))
        : range(pcount(sim.colloid), 1, step=-1)
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

function apply_translation!(sim::ColloidSim, randnums::Matrix{<:Real}, idx::Int)
    r = sim.move_radius * randnums[1, idx]
    θ = 2π * randnums[2, idx]
    x, y = r * cos(θ), r * sin(θ)

    i, j = get_cell_list_indices(sim.colloid, sim.cell_list, idx)
    deleteat!(sim.cell_list.cells[i, j], findfirst(==(idx), sim.cell_list.cells[i, j]))
    move!(sim.colloid, idx, x, y)
    i, j = get_cell_list_indices(sim.colloid, sim.cell_list, idx)
    push!(sim.cell_list.cells[i, j], idx)

    if has_violation(sim, randnums, idx, i, j)
        move!(sim.colloid, idx, -x, -y)
        pop!(sim.cell_list.cells[i, j])
        i, j = get_cell_list_indices(sim.colloid, sim.cell_list, idx)
        push!(sim.cell_list.cells[i, j], idx)
        sim.rejected_translations += 1
    else
        sim.accepted_translations += 1
    end
end

function apply_rotation!(sim::ColloidSim,randnums::Matrix{<:Real}, idx::Int)
    angle_change = sim.rotation_span * (randnums[2, idx] - 0.5)
    sim.colloid.angles[idx] += angle_change
    i, j = get_cell_list_indices(sim.colloid, sim.cell_list, idx)
    if has_violation(sim, randnums, idx, i, j)
        sim.colloid.angles[idx] -= angle_change
        sim.rejected_rotations += 1
    else
        sim.accepted_rotations += 1
    end
end

@inline function has_violation(sim::ColloidSim, randnums::Matrix{<:Real},
                               idx::Integer, i::Integer, j::Integer)
    if !isnothing(sim.potential) || !isnothing(sim.pairpotential)
        potsum = zero(sim.particle_potentials)
        if !isnothing(sim.pairpotential)
            potsum += pairpotential_sum(
                sim.colloid, sim.cell_list, sim.pairpotential, idx, i, j)
        end
        if !isnothing(sim.potential)
            potsum += sim.potential(sim.colloid, idx)
        end
        potchange = potsum - sim.particle_potentials[idx]
        return (violates_constraints(sim, idx)
            || randnums[3, idx] > exp(-sim.beta * potchange))
    else
        return (violates_constraints(sim, idx)
            || has_overlap(sim.colloid, sim.cell_list, idx, i, j))
    end
end

@inline function violates_constraints(sim::ColloidSim, idx::Integer)
    for constraint in sim.constraints
        if is_violated(sim.colloid, constraint, idx)
            return true
        end
    end
    return false
end
