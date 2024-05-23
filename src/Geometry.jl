#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/20 14:57:59
  @ license: MIT
  @ description: the basic geometry, wait for further development
 =#

abstract type Shape end

struct CalculationDomainBoundingBox <: Shape
    start_point_::RealVector
    end_point_::RealVector
end

Base.show(io::IO, shape::CalculationDomainBoundingBox) = print(
    io,
    "CalculationDomainBoundingBox:\n",
    "    start from: $(shape.start_point_)\n",
    "    end at    : $(shape.end_point_)\n",
)

@inline function isInsideShape(x::RealVector, shape::CalculationDomainBoundingBox)::Bool
    @inbounds return (
        shape.start_point_[1] <= x[1] <= shape.end_point_[1] &&
        shape.start_point_[2] <= x[2] <= shape.end_point_[2] &&
        shape.start_point_[3] <= x[3] <= shape.end_point_[3]
    )
end
