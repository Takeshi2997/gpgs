include("./setup.jl")
include("./model.jl")
using Base.Threads, Serialization, LinearAlgebra

const filenames = ["gpdata" * lpad(it, 4, "0") * ".dat" for it in 0:c.iT]
const filename  = "physicalvalue.txt"

function imaginarytime(model::GPmodel)
    for it in 1:c.iT
        # Model Update!
        xs, ys = model.xs, model.ys
        ys′ = copy(ys)
        @simd for i in 1:c.num
            x = xs[i]
            y = ys[i]
            e = energy(x, y, model)
            ys′[i] = log((1f0 - c.Δτ * e / c.nspin) * exp(y))
        end
        v = sum(exp.(ys′)) / c.ndata
        ys′ .-= v
        model = makemodel(xs, ys′)
        outdata = (xs, ys′)
        open(io -> serialize(io, outdata), "./data/" * filenames[it+1], "w")
    end
end

