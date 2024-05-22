#=
  @ author: bcynuaa <bcynuaa@163.com> | callm1101 <Calm.Liu@outlook.com>
  @ date: 2024/05/21 17:25:16
  @ license: MIT
  @ description:
 =#

abstract type SmoothKernel end

# * CubicSpline kernel
struct CubicSpline <: SmoothKernel
    h_::Float64
    dim_::Int64
    radius_ratio_::Float64
    influence_radius_::Float64
    sigma_::Float64
    kernel_0_::Float64
end

# * CubicSpline kernel value
function kernelValue(r::Float64, kernel::CubicSpline)::Float64
    q::Float64 = r / kernel.h_
    if q < 1.0
        return kernel.sigma_ / 4.0 * (3.0 * q^2 * (q - 2.0) + 4.0)
    elseif q < 2.0
        return kernel.sigma_ / 4.0 * (2.0 - q)^3
    else
        return 0.0
    end
end

# * CubicSpline kernel gradient
function kernelGradient(r::Float64, kernel::CubicSpline)::Float64
    q::Float64 = r / kernel.h_
    if q < 1.0
        return kernel.sigma_ / kernel.h_ * 3.0 / 4.0 * q * (3.0 * q - 4.0)
    elseif q < 2.0
        return -kernel.sigma_ / kernel.h_ * 3.0 / 4.0 * (2.0 - q)^2
    else
        return 0.0
    end
end

# * Gaussian kernel
struct Gaussian <: SmoothKernel
    h_::Float64
    dim_::Int64
    radius_ratio_::Float64
    influence_radius_::Float64
    sigma_::Float64
    kernel_0_::Float64
end

# * Gaussian kernel value
function kernelValue(r::Float64, kernel::Gaussian)::Float64
    q::Float64 = r / kernel.h_
    if q < 3.0
        return kernel.sigma_ * exp(-q^2)
    else
        return 0.0
    end
end

function kernelGradient(r::Float64, kernel::Gaussian)::Float64
    q::Float64 = r / kernel.h_
    if q < 3.0
        return -2.0 * kernel.sigma_ / kernel.h_ * q * exp(-q^2)
    else
        return 0.0
    end
end

# * WendlandC2 kernel
struct WendlandC2 <: SmoothKernel
    h_::Float64
    dim_::Int64
    radius_ratio_::Float64
    influence_radius_::Float64
    sigma_::Float64
    kernel_0_::Float64
end

# * WendlandC2 kernel value
function kernelValue(r::Float64, kernel::WendlandC2)::Float64
    q::Float64 = r / kernel.h_
    if q < 2.0
        return kernel.sigma_ * (2.0 - q)^4 * (1.0 + 2.0 * q) / 16.0
    else
        return 0.0
    end
end

# * WendlandC2 kernel gradient
function kernelGradient(r::Float64, kernel::WendlandC2)::Float64
    q::Float64 = r / kernel.h_
    if q < 2.0
        return -kernel.sigma_ / kernel.h_ * 5.0 / 8.0 * q * (2.0 - q)^3
    else
        return 0.0
    end
end

# * WendlandC4 kernel
struct WendlandC4 <: SmoothKernel
    h_::Float64
    dim_::Int64
    radius_ratio_::Float64
    influence_radius_::Float64
    sigma_::Float64
    kernel_0_::Float64
end

# * WendlandC4 kernel value
function kernelValue(r::Float64, kernel::WendlandC4)::Float64
    q::Float64 = r / kernel.h_
    if q < 2.0
        return kernel.sigma_ * (2.0 - q)^6 * (35.0 * q^2 + 36.0 * q + 12.0) / 768.0
    else
        return 0.0
    end
end

# * WendlandC4 kernel gradient
function kernelGradient(r::Float64, kernel::Float64)::Float64
    q::Float64 = r / kernel.h_
    if q < 2.0
        return -kernel.sigma_ / kernel.h_ * (35.0 / 96.0 * q^2 + 7.0 / 48.0 * q) * (2.0 - q)^5
    else
        return 0.0
    end
end

const kSmoothKernelParametersDict = Dict(
    CubicSpline => (2.0, [2.0 / 3.0, 10.0 / 7.0 / pi, 1.0 / pi]),
    Gaussian => (3.0, [1.0 / sqrt(pi), 1.0 / pi, 1.0 / sqrt(pi^3)]),
    WendlandC2 => (2.0, [0.0, 7.0 / 4.0 / pi, 21.0 / 16.0 / pi]),
    WendlandC4 => (2.0, [5.0 / 8.0, 9.0 / 4.0 / pi, 495.0 / 256.0 / pi]),
)

@inline function SmoothKernel(influence_radius::Float64, dim::Int64, SmoothKernelType::DataType)::SmoothKernelType
    @assert(dim in [1, 2, 3], "The dimension of the space must be 1, 2, or 3.")
    radius_ratio = kSmoothKernelParametersDict[SmoothKernelType][1]
    h = influence_radius / radius_ratio
    sigma = kSmoothKernelParametersDict[SmoothKernelType][2][dim] / h^dim
    kernel_0 = sigma
    return SmoothKernelType(h, dim, radius_ratio, influence_radius, sigma, kernel_0)
end
