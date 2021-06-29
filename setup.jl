struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    H::T
    Δτ::T
    A::T
end
function GP_Data()
    NSpin = 80
    NData = 64
    NMC = 1024
    MCSkip = 16
    H = 4.0
    Δτ = 0.1
    A = 0.1
    GP_Data(NSpin, NData, NMC, MCSkip, H, Δτ, A)
end
c = GP_Data()


