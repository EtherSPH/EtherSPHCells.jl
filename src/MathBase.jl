#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/19 16:34:08
  @ license: MIT
  @ description: the basic math functions and containers
 =#

const IntegerVector = SVector{3, Int64}

"""
    RealVector
    specialized 3D vector type
"""
const RealVector = SVector{3, Float64}

const kVecX = RealVector(1.0, 0.0, 0.0)
const kVecY = RealVector(0.0, 1.0, 0.0)
const kVecZ = RealVector(0.0, 0.0, 1.0)
const kVec0 = RealVector(0.0, 0.0, 0.0)

@inline function Base.:+(x::RealVector, y::RealVector)::RealVector
    @inbounds return RealVector(x[1] + y[1], x[2] + y[2], x[3] + y[3])
end

@inline function Base.:-(x::RealVector, y::RealVector)::RealVector
    @inbounds return RealVector(x[1] - y[1], x[2] - y[2], x[3] - y[3])
end

@inline function Base.:*(x::RealVector, y::RealVector)::RealVector
    @inbounds return RealVector(x[1] * y[1], x[2] * y[2], x[3] * y[3])
end

@inline function Base.:*(x::RealVector, y::Float64)::RealVector
    @inbounds return RealVector(x[1] * y, x[2] * y, x[3] * y)
end

@inline function dot(x::RealVector, y::RealVector)::Float64
    @inbounds return x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
end

@inline function norm(x::RealVector)::Float64
    return sqrt(dot(x, x))
end

@inline function cross(x::RealVector, y::RealVector)::RealVector
    @inbounds return RealVector(x[2] * y[3] - x[3] * y[2], x[3] * y[1] - x[1] * y[3], x[1] * y[2] - x[2] * y[1])
end

"""
    RealMatrix
    specialized 3x3 matrix type
"""

const RealMatrix = SMatrix{3, 3, Float64, 9}

const kMat0 = RealMatrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
const kMatI = RealMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

@inline function trace(m::RealMatrix)::Float64
    @inbounds return m[1] + m[5] + m[9]
end

@inline function dev(m::RealMatrix)::RealMatrix
    return m - trace(m) / 3 * kMatI
end

@inline function det(m::RealMatrix)::Float64
    @inbounds return (
        +m[1] * m[5] * m[9] + m[2] * m[6] * m[7] + m[3] * m[4] * m[8] - m[7] * m[5] * m[3] - m[8] * m[6] * m[1] -
        m[9] * m[4] * m[2]
    )
end

@inline function transpose(m::RealMatrix)::RealMatrix
    @inbounds return RealMatrix(m[1], m[4], m[7], m[2], m[5], m[8], m[3], m[6], m[9])
end

@inline function cofactor(m::RealMatrix)::RealMatrix
    @inbounds return RealMatrix(
        +m[5] * m[9] - m[6] * m[8],
        -m[4] * m[9] + m[6] * m[7],
        +m[4] * m[8] - m[5] * m[7],
        -m[2] * m[9] + m[3] * m[8],
        +m[1] * m[9] - m[3] * m[7],
        -m[1] * m[8] + m[2] * m[7],
        +m[2] * m[6] - m[3] * m[5],
        -m[1] * m[6] + m[3] * m[4],
        +m[1] * m[5] - m[2] * m[4],
    )
end

@inline function inv(m::RealMatrix)::RealMatrix
    return 1 / det(m) * transpose(cofactor(m))
end

@inline function ddot(x::RealMatrix, y::RealMatrix)::Float64
    @inbounds return (
        +x[1] * y[1] +
        x[2] * y[2] +
        x[3] * y[3] +
        x[4] * y[4] +
        x[5] * y[5] +
        x[6] * y[6] +
        x[7] * y[7] +
        x[8] * y[8] +
        x[9] * y[9]
    )
end
