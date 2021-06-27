include("./setup.jl")
include("./functions.jl")
include("./model.jl")
using Base.Threads, Serialization, LinearAlgebra

const filenames = ["gpdata" * lpad(it, 4, "0") * ".dat" for it in 0:c.iT]
const filename  = "physicalvalue.txt"

function imaginarytime(model::GPmodel)
    for it in 1:c.iT
        # Model Update!
        xs, ys = model.xs, model.ys
        ys′ = copy(ys)
        @simd for i in 1:c.ndata
            x = xs[i]
            y = ys[i]
            e = energy(x, y, model)
            ys′[i] = log((1.0 - c.Δτ * e / c.nspin) * exp(y))
        end
        v = sum(exp.(ys′)) / c.ndata
        ys′ .-= v
        model = GPmodel(xs, ys′)
        x_spins = [x.spin for x in xs]
        outdata = (x_spins, ys′)
        open(io -> serialize(io, outdata), "./data/" * filenames[it+1], "w")
    end
end

function measure()
    touch("./data/" * filename)
    # Imaginary roop
    for it in 1:c.iT
        # Make model
        xydata = open(deserialize, "./data/" * filenames[it+1])
        x_spins, ys = xydata
        xs = Vector{State}(undef, c.ndata)
        for i in 1:c.ndata
            xs[i] = State(x_spins[i])
        end
        model = GPmodel(xs, ys)

        # Metropolice sampling
        x_mc = mh(model)

        # Calculate Physical Value
        u = 0.0
        m = 0.0
        @threads for x in x_mc
            y = predict(model, x)
            e = energy(x, y, model) / c.nspin
            h = sum(@views x[1:c.nspin]) / c.nspin
            u += e
            m += h
        end
        ene = real(u) / c.nmc
        mag = m / c.nmc

        # Write Data
        open("./data/" * filename, "a") do io
            write(io, string(it))
            write(io, "\t")
            write(io, string(ene))
            write(io, "\t")
            write(io, string(mag))
            write(io, "\n")
        end
    end 
end

function mh(model::GPmodel)
    outxs = Vector{State}(undef, c.nmc)
    x = State(rand([1f0, -1f0], c.nspin))
    @inbounds for i in 1:c.nmc
        for j in 1:c.mcskip
            update!(model, x)
        end
        outxs[i] = x
    end
    outxs
end
