struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    H::T
    l::T
    A::T
end
function GP_Data()
    NSpin = 80
    NData = 64
    NMC = 1024
    MCSkip = 16
    H = 4.0
    l = 0.6
    A = 0.4
    GP_Data(NSpin, NData, NMC, MCSkip, H, l, A)
end
c = GP_Data()


