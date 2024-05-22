#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/20 21:28:57
  @ license: MIT
  @ description:
 =#

const box2d = CalculationDomainBoundingBox(RealVector(2.2, 1.3, 0.0), RealVector(4.3, 2.7, 0.0))

const h = 0.15

const bcl2d = BackgroundCellList(h, box2d)

@test getCartesian3DIndexInBackgroundCellList(RealVector(2.25, 1.35, 0.0), bcl2d) == IntegerVector(1, 1, 1)
@test getCartesian3DIndexInBackgroundCellList(RealVector(4.3, 2.7, 0.0), bcl2d) == IntegerVector(14, 10, 1)
@test getCartesian3DIndexInBackgroundCellList(RealVector(2.8, 1.35, 0.0), bcl2d) == IntegerVector(4, 1, 1)
@test getCartesian3DIndexInBackgroundCellList(RealVector(2.2, 1.8, 0.0), bcl2d) == IntegerVector(1, 4, 1)

@test getGlobalIndexInBackgroundCellList(RealVector(2.25, 1.35, 0.0), bcl2d) == 1
@test getGlobalIndexInBackgroundCellList(RealVector(4.3, 2.7, 0.0), bcl2d) == 140
@test getGlobalIndexInBackgroundCellList(RealVector(2.8, 1.35, 0.0), bcl2d) == 4
@test getGlobalIndexInBackgroundCellList(RealVector(2.2, 1.8, 0.0), bcl2d) == 43
