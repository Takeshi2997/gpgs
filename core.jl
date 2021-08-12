include("./setup.jl")
include("./model.jl")
using Base.Threads, LinearAlgebra, Random, Folds

function imaginarytime(model::GPmodel)
    data_x, data_y = model.data_x, model.data_y
    ψ = copy(data_y)
    @threads for i in 1:c.NData
        e = localenergy(data_x[i], model)
        ψ[i] = (c.l - e / c.NSpin) * exp(data_y[i])
    end
    data_y = log.(ψ)
    # v = sum(ψ) / c.NData
    # data_y .-= log(v)
    GPmodel(data_x, data_y)
end

function tryflip(x::State, model::GPmodel, eng::MersenneTwister)
    pos = rand(eng, collect(1:c.NSpin))
    y = predict(x, model)
    xflip_spin = copy(x.spin)
    xflip_spin[pos] *= -1
    xflip = State(xflip_spin)
    y_new = predict(xflip, model)
    x.spin[pos] *= ifelse(rand(eng) < exp(2 * real(y_new - y)), -1, 1)
    State(x.spin)
end

function localenergy(x::State, model::GPmodel)
    y = predict(x, model)
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        xflip_spin = copy(x.spin)
        xflip_spin[i] *= -1
        xflip = State(xflip_spin)
        y2 = predict(xflip, model)
        e = -x.spin[i] * x.spin[i%c.NSpin+1] - c.H * exp(y2 - y)
        eloc += e
    end
    eloc
end

function physicalvals(x::State, model::GPmodel)
    y = predict(x, model)
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        xflip_spin = copy(x.spin)
        xflip_spin[i] *= -1
        xflip = State(xflip_spin)
        y2 = predict(xflip, model)
        e = -x.spin[i] * x.spin[i%c.NSpin+1] - c.H * exp(y2 - y)
        eloc += e
    end
    eloc
end

function energy(x_mc::Vector{State}, model::GPmodel)
    @threads for i in 1:c.NMC
        @simd for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            x_mc[i] = tryflip(x_mc[i], model, eng)
        end
    end
    ene = Folds.sum(physicalvals(x, model) for x in x_mc)
    real(ene / c.NMC)
end

