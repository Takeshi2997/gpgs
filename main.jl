using LinearAlgebra, Random, Base.Threads

struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    H::T
end
function GP_Data()
    NSpin = 80
    NData = 64
    NMC = 1024
    MCSkip = 16
    H = 4.0
    GP_Data(NSpin, NData, NMC, MCSkip, H)
end
const c = GP_Data()

EngArray = [MersenneTwister(1234) for i in 1:nthreads()]

mutable struct State{T<:Real}
    spin::Vector{T}
    shift::Vector{Vector{T}}
end
function State(x::Vector{T}) where {T<:Real}
    shift = [circshift(x, s) for s in 1:c.NSpin]
    State(x, shift)
end

function choosesample(data_x::Vector{State}, data_y::Vector{T}) where {T<:Real}
    for i in 1:length(data_y)
        x = rand([1.0, -1.0], c.NSpin)
        data_x[i] = State(x)
        data_y[i] = rand()
    end
end

const A = 100

function kernel(x1::State, x2::State)
    v = [norm(x1.shift[n] - x2.spin) for n in 1:length(x1.spin)]
    v ./= c.NSpin
    sum(exp.(-v ./ A))
end

function makeinverse(KI::Array{T}, data_x::Vector{State}) where {T<:Real}
    for i in 1:c.NData
        for j in i:c.NData
            KI[i, j] = kernel(data_x[i], data_x[j])
            KI[j, i] = KI[i, j]
        end
    end
    U, Δ, V = svd(KI)
    invΔ = Diagonal(1f0 ./ Δ .* (Δ .> 1f-6))
    KI[:, :] = V * invΔ * U'
end

function makepvector(data_x::Vector{State}, data_y::Vector{T}, pvec::Vector{T}) where {T<:Real}
    KI = Array{T}(undef, c.NData, c.NData)
    makeinverse(KI, data_x)
    pvec[:] = KI * data_y
end

function predict(x::State, data_x::Vector{State}, pvector::Vector{T}) where {T<:Real}
    kv = [kernel(x, data_x[i]) for i in 1:c.NData]
    kv' * pvector
end

function imaginarytime(data_x::Vector{State}, data_y::Vector{T}, pvector::Vector{T}) where {T<:Real}
    H_ψ = zeros(T, c.NData)
    for i in 1:c.NData
        for j in 1:c.NData
            H_ψ[i] -= data_x[i].spin[j] * data_x[i].spin[j%c.NSpin+1] * exp(data_y[i])
            xtmp_spin = copy(data_x[i].spin)
            xtmp = State(xtmp_spin)
            y = predict(xtmp, data_x, pvector)
            H_ψ[i] -= c.H * exp(y)
        end
    end
    data_y = exp.(data_y) - H_ψ * 0.1
    v = sum(exp.(data_y)) / c.NData
    data_y .-= log.(v)
end

function tryflip(x::State, data_x::Vector{State}, pvector::Vector{T}, eng::MersenneTwister) where {T<:Real}
    pos = rand(eng, collect(1:c.NSpin))
    y = predict(x, data_x, pvector)
    x.spin[pos] *= -1
    y_new = predict(x, data_x, pvector)
    x.spin[pos] = ifelse(rand(eng) > exp(2 * (y_new - y)), -1.0, 1.0)
end

function localenergy(x::State, data_x::Vector{State}, pvector::Vector{T}) where {T<:Real}
    y = predict(x, data_x, pvector)
    eloc = 0.0
    for i in 1:c.NSpin
        eloc -= x.spin[i] * x.spin[i%c.NSpin+1]
        x.spin[i] *= -1
        y2 = predict(x, data_x, pvector)
        eloc -= c.H * exp(y2 - y)
        x.spin[i] *= -1
    end
    eloc
end

function energy(x_mc::Vector{State}, data_x::Vector{State}, pvector::Vector{T}) where {T<:Real}
    @threads for i in 1:c.NMC
        for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            tryflip(x_mc[i], data_x, pvector, eng)
        end
    end
    ene = 0.0
    @threads for i in 1:c.NMC
        ene += localenergy(x_mc[i], data_x, pvector)
    end
    ene / c.NMC
end

function main()
    for i in 1:nthreads()
        EngArray[i] = MersenneTwister(i+0)
    end
    eng = EngArray[1]
    data_x = Vector{State}(undef, c.NData)
    data_y = Vector{Float64}(undef, c.NData)
    choosesample(data_x, data_y)

    batch_x = Vector{State}(undef, c.NMC)
    for i in 1:c.NMC
        x = rand(eng, [1.0, -1.0], c.NSpin)
        batch_x[i] = State(x)
    end
    
    for i in 1:200
        pvec = Vector{Float64}(undef, c.NData)
        makepvector(data_x, data_y, pvec)
        imaginarytime(data_x, data_y, pvec)
        ene = energy(batch_x, data_x, pvec)
        open("./data/" * filename, "a") do io
            write(io, string(i))
            write(io, "\t")
            write(io, string(ene))
            write(io, "\n")
        end
    end
end

main()