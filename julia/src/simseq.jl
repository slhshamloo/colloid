function apply_step!(sim::Simulation, cell_list::SeqCellList)
    randchoices = rand(Bool, particle_count(sim.colloid))
    randnums = rand(sim.numtype, 2, particle_count(sim.colloid))
    iter = (rand(Bool) ?
        range(1, particle_count(sim.colloid))
        : range(particle_count(sim.colloid), 1, step=-1)
    )
    for idx in iter
        if randchoices[idx]
            apply_translation!(sim, cell_list, randnums, idx)
        else
            apply_rotation!(sim, cell_list, randnums, idx)
        end
    end
    return true
end

function apply_translation!(sim::Simulation, cell_list::SeqCellList,
                            randnums::Matrix{<:Real}, idx::Int)
    r = sim.move_radius * randnums[1, idx]
    θ = 2π * randnums[2, idx]
    x, y = r * cos(θ), r * sin(θ)

    i, j = get_cell_list_indices(sim.colloid, cell_list, idx)
    deleteat!(cell_list.cells[i, j], findfirst(==(idx), cell_list.cells[i, j]))
    move!(sim.colloid, idx, x, y)
    i, j = get_cell_list_indices(sim.colloid, cell_list, idx)
    push!(cell_list.cells[i, j], idx)

    if violates_constraints(sim, idx) || has_overlap(sim.colloid, cell_list, idx, i, j)
        move!(sim.colloid, idx, -x, -y)
        pop!(cell_list.cells[i, j])
        i, j = get_cell_list_indices(sim.colloid, cell_list, idx)
        push!(cell_list.cells[i, j], idx)
        sim.rejected_translations += 1
    else
        sim.accepted_translations += 1
    end
end

function apply_rotation!(sim::Simulation, cell_list::SeqCellList,
                         randnums::Matrix{<:Real}, idx::Int)
    angle_change = sim.rotation_span * (randnums[2, idx] - 0.5)
    sim.colloid.angles[idx] += angle_change
    i, j = get_cell_list_indices(sim.colloid, cell_list, idx)
    if has_overlap(sim.colloid, cell_list, idx, i, j)
        sim.colloid.angles[idx] -= angle_change
        sim.rejected_rotations += 1
    else
        sim.accepted_rotations += 1
    end
end

@inline function violates_constraints(sim::Simulation, idx::Integer)
    for constraint in sim.constraints
        if is_violated(sim, constraint, idx)
            return true
        end
    end
    return false
end
