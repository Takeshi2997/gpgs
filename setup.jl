struct GP_Data{T<:Real, S<:Integer}
    # System Size
    nspin::S
    
    # System Param
    h::T
    J::T
    t::T

    # Update Param
    Δτ::T
    
    # Repeat Number
    ndata::S
    nmc::S
    mcskip::S
    iT::S
    
    # Hyper Param
    θ₁::T
    θ₂::T
end

function GP_Data()
    nspin = 80
    h = 1.0
    J = 1.0
    t = 1.0
    Δτ = 0.1
    ndata = 64
    nmc = 1024
    mcskip = 16
    batchsize = 8
    iT = 200
    θ₁ = 1.0
    θ₂ = 100.0
    GP_Data(nspin, h, J, t, Δτ, ndata, nmc, mcskip, iT, θ₁, θ₂)
end

c = GP_Data()
