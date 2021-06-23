struct GP_Data{T<:Real, S<:Integer}
    # System Size
    N::S
    
    # System Param
    h::T
    J::T
    t::T
    Δτ::T
    
    # Repeat Number
    num::S
    auxn::S
    burnintime::S
    iters::S
    batchsize::S
    iT::S
    
    # Hyper Param
    θ₁::T
    θ₂::T
end

function GP_Data()
    N = 80
    h = 1f0
    J = 1f0
    t = 1f0
    Δτ = 0.1f0
    num = 1024
    auxn = 64
    burnintime = 10
    iters = 1000
    batchsize = 8
    iT = 100
    θ₁ = 1f0
    θ₂ = 0.01f0
    GP_Data(N, h, J, t, Δτ, num, auxn, burnintime, iters, batchsize, iT, θ₁, θ₂)
end

const c = GP_Data()
