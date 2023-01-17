function apply_step!(sim::Simulation, cell_list::CuCellList)
    nthreads = (numthreads[1] * numthreads[2])
    sweeps = ceil(Int, mean(cell_list.counts))
    randnums = CUDA.rand(sim.numtype, 3, size(cell_list.counts, 1),
                         size(cell_list.counts, 2), sweeps)
    randchoices = CUDA.rand(Bool, size(cell_list.counts, 1),
                            size(cell_list.counts, 2), sweeps)
    for sweep in 1:sweeps
        for color in shuffle(1:4)
            cellcount = size(cell_list.checkerboard[color], 2)
            numblocks = cellcount ÷ nthreads + 1
            @cuda threads=nthreads blocks=numblocks apply_parallel_step!( #=
                =# sim.colloid, cell_list, color, sweep, sim.move_radius, #=
                =# sim.rotation_span, randnums, randchoices)
        end
        direction = ((1, 0), (-1, 0), (0, 1), (0, -1))[rand(1:4)]
        shift = (direction[2] == 0 ? cell_list.width[1] : cell_list.width[2]) * (
            rand(sim.numtype) / 2)
        shift_cells!(sim.colloid, cell_list, direction, shift)
    end
end

function apply_parallel_step!(colloid::Colloid, cell_list::CuCellList,
        color::Integer, sweep::Integer, move_radius::Real, rotation_span::Real,
        randnums::CuDeviceArray, randchoices::CuDeviceArray)
    cell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    colorgroup = cell_list.checkerboard[color]
    if cell > size(colorgroup, 2)
        return
    end

    i, j = colorgroup[1, cell], colorgroup[2, cell]
    ncell = cell_list.counts[i, j]
    if ncell == 0
        return
    end
    idx = cell_list.cells[ceil(Int, randnums[1, i, j, sweep] * ncell), i, j]

    if randchoices[i, j, sweep]
        apply_translation!(colloid, cell_list, randnums, move_radius, sweep, i, j, idx)
    else
        apply_rotation!(colloid, cell_list, randnums, rotation_span, sweep, i, j, idx)
    end
    return
end

function apply_translation!(colloid::Colloid, cell_list::CuCellList,
                            randnums::CuDeviceArray, move_radius::Real,
                            sweep::Integer, i::Int32, j::Int32, idx::Int32)
    r = move_radius * randnums[2, i, j, sweep]
    θ = 2π * randnums[3, i, j, sweep]
    x, y = r * cos(θ), r * sin(θ)
    move!(colloid, idx, x, y)
    if ((i, j) != get_cell_list_indices(colloid, cell_list, idx)
            # || violates_constraints(colloid, idx)
            || has_overlap(colloid, cell_list, idx, i, j))
        move!(colloid, idx, -x, -y)
    else

    end
end

function apply_rotation!(colloid::Colloid, cell_list::CuCellList,
                         randnums::CuDeviceArray, rotation_span::Real,
                         sweep::Integer, i::Int32, j::Int32, idx::Int32)
    angle_change = rotation_span * (randnums[2, i, j, sweep] - 0.5)
    colloid.angles[idx] += angle_change
    if (#= violates_constraints(colloid, idx)
            || =# has_overlap(colloid, cell_list, idx, i, j))
        colloid.angles[idx] -= angle_change
    else

    end
end
