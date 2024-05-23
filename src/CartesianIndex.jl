#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/20 22:12:38
  @ license: MIT
  @ description: 3d cartesian index and range used in the cell list neighbour search
 =#

@inline function getCartesian3DIndexFromGlobalIndex(
    global_index::Int64,
    cartesian_3d_range_index::IntegerVector,
)::IntegerVector
    @inbounds k = cld(global_index, cartesian_3d_range_index[1] * cartesian_3d_range_index[2])
    @inbounds j = cld(
        global_index - (k - 1) * cartesian_3d_range_index[1] * cartesian_3d_range_index[2],
        cartesian_3d_range_index[1],
    )
    @inbounds i =
        global_index - (k - 1) * cartesian_3d_range_index[1] * cartesian_3d_range_index[2] -
        (j - 1) * cartesian_3d_range_index[1]
    return IntegerVector(i, j, k)
end

@inline function getGlobalIndexFromCartesian3DIndex(
    cartesian_3d_index::IntegerVector,
    cartesian_3d_range_index::IntegerVector,
)::Int64
    @inbounds return (
        cartesian_3d_index[1] +
        (cartesian_3d_index[2] - 1) * cartesian_3d_range_index[1] +
        (cartesian_3d_index[3] - 1) * cartesian_3d_range_index[1] * cartesian_3d_range_index[2]
    )
end

@inline function isCartesian3DIndexInCartesian3DRange(
    cartesian_3d_index::IntegerVector,
    cartesian_3d_range_index::IntegerVector,
)::Bool
    @inbounds return (
        1 <= cartesian_3d_index[1] <= cartesian_3d_range_index[1] &&
        1 <= cartesian_3d_index[2] <= cartesian_3d_range_index[2] &&
        1 <= cartesian_3d_index[3] <= cartesian_3d_range_index[3]
    )
end

@inline function isGlobalIndexInCartesian3DRange(global_index::Int64, cartesian_3d_range_index::IntegerVector)::Bool
    return isCartesian3DIndexInCartesian3DRange(
        getCartesian3DIndexFromGlobalIndex(global_index, cartesian_3d_range_index),
        cartesian_3d_range_index,
    )
end
