#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/21 00:46:17
  @ license: MIT
  @ description: the background cell list used in the particle system
 =#

mutable struct BackgroundCellList
    reference_smoothing_radius_::Float64
    calculation_domain_bounding_box_::CalculationDomainBoundingBox
    restrict_calculation_domain_::Shape
    cartesian_3d_start_index_::IntegerVector
    cartesian_3d_end_index_::IntegerVector
    cartesian_3d_range_index_::IntegerVector
    cell_number_::Int64
    cell_list_::Vector{Cell}
    to_be_removed_cell_::Cell # which used to contain the particles to be removed
end

Base.show(io::IO, background_cell_list::BackgroundCellList) = print(
    io,
    "Background Cell List:\n",
    "    reference smoothing length: $(background_cell_list.reference_smoothing_radius_)\n",
    "    calculation domain bounding box:\n",
    "        start from: $(background_cell_list.calculation_domain_bounding_box_.start_point_)\n",
    "        end at    : $(background_cell_list.calculation_domain_bounding_box_.end_point_)\n",
    "    cartesian 3d index:\n",
    "        start: $(background_cell_list.cartesian_3d_start_index_)\n",
    "        range: $(background_cell_list.cartesian_3d_range_index_)\n",
    "        end  : $(background_cell_list.cartesian_3d_end_index_)\n",
    "    cell list number: $(background_cell_list.cell_number_)\n",
)

function BackgroundCellList(
    reference_smoothing_radius::Float64,
    calculation_domain_bounding_box::CalculationDomainBoundingBox,
    restrict_calculation_domain::Shape,
)::BackgroundCellList
    x_min = calculation_domain_bounding_box.start_point_
    x_max = calculation_domain_bounding_box.end_point_
    cartesian_3d_start_index = cld.(x_min, reference_smoothing_radius) .|> Int64
    cartesian_3d_end_index = ceil.(x_max ./ reference_smoothing_radius) .|> Int64
    cartesian_3d_range_index_list = []
    for i in eachindex(cartesian_3d_start_index)
        push!(cartesian_3d_range_index_list, max(1, cartesian_3d_end_index[i] - cartesian_3d_start_index[i]))
    end
    cartesian_3d_range_index = IntegerVector(cartesian_3d_range_index_list)
    cell_number = prod(cartesian_3d_range_index)
    cell_list::Vector{Cell} = Cell[Cell() for _ in 1:cell_number]
    to_be_removed_cell = Cell()
    if cartesian_3d_range_index[3] == 1
        # for 2D background cell list build
        Threads.@threads for current_cell_id in eachindex(cell_list)
            neighbour_cell_global_ids = []
            current_cell_cartesian_index = getCartesian3DIndexFromGlobalIndex(current_cell_id, cartesian_3d_range_index)
            for i in -1:1, j in -1:1
                neighbour_cell_cartesian_index = current_cell_cartesian_index + IntegerVector(i, j, 0)
                if isCartesian3DIndexInCartesian3DRange(neighbour_cell_cartesian_index, cartesian_3d_range_index)
                    neighbour_cell_global_id =
                        getGlobalIndexFromCartesian3DIndex(neighbour_cell_cartesian_index, cartesian_3d_range_index)
                    push!(neighbour_cell_global_ids, neighbour_cell_global_id)
                end
            end
            cell_list[current_cell_id].neighbour_cell_global_ids_ = neighbour_cell_global_ids
        end
    else
        # for 3D background cell list build
        Threads.@threads for current_cell_id in eachindex(cell_list)
            neighbour_cell_global_ids = []
            current_cell_cartesian_index = getCartesian3DIndexFromGlobalIndex(current_cell_id, cartesian_3d_range_index)
            for i in -1:1, j in -1:1, k in -1:1
                neighbour_cell_cartesian_index = current_cell_cartesian_index + IntegerVector(i, j, k)
                if isCartesian3DIndexInCartesian3DRange(neighbour_cell_cartesian_index, cartesian_3d_range_index)
                    neighbour_cell_global_id =
                        getGlobalIndexFromCartesian3DIndex(neighbour_cell_cartesian_index, cartesian_3d_range_index)
                    push!(neighbour_cell_global_ids, neighbour_cell_global_id)
                end
            end
            cell_list[current_cell_id].neighbour_cell_global_ids_ = neighbour_cell_global_ids
        end
    end
    return BackgroundCellList(
        reference_smoothing_radius,
        calculation_domain_bounding_box,
        restrict_calculation_domain,
        cartesian_3d_start_index,
        cartesian_3d_end_index,
        cartesian_3d_range_index,
        cell_number,
        cell_list,
        to_be_removed_cell,
    )
end

function BackgroundCellList(
    reference_smoothing_radius::Float64,
    calculation_domain_bounding_box::CalculationDomainBoundingBox,
)::BackgroundCellList
    restrict_calculation_domain = calculation_domain_bounding_box
    return BackgroundCellList(reference_smoothing_radius, calculation_domain_bounding_box, restrict_calculation_domain)
end

@inline function getCartesian3DIndexInBackgroundCellList(
    position::RealVector,
    background_cell_list::BackgroundCellList,
)::IntegerVector
    return max.(
        cld.(
            position .- background_cell_list.calculation_domain_bounding_box_.start_point_,
            background_cell_list.reference_smoothing_radius_,
        ) .|> Int64,
        1,
    )
end

@inline function getGlobalIndexInBackgroundCellList(
    position::RealVector,
    background_cell_list::BackgroundCellList,
)::Int64
    return getGlobalIndexFromCartesian3DIndex(
        getCartesian3DIndexInBackgroundCellList(position, background_cell_list),
        background_cell_list.cartesian_3d_range_index_,
    )
end

@inline function isInBackgroundCellList(position::RealVector, background_cell_list::BackgroundCellList)::Bool
    return isInsideShape(position, background_cell_list.restrict_calculation_domain_)
end
