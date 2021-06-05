include("./setup.jl")
include("./functions.jl")
include("./model.jl")
using Distributions, Base.Threads, Serialization, LinearAlgebra

const filenames = ["gpdata" * lpad(it, 4, "0") * ".dat" for it in 0:c.iT]
const filename  = "physicalvalue.txt"

function it_evolution(model::GPmodel)
    for it in 1:c.iT
        # Model Update!
        xydata = open(deserialize, "./data/" * filenames[it])
        xs, ys = model.xs, model.ys
        ys′ = copy(ys)
        @simd for i in 1:c.num
            x = xs[i]
            y = ys[i]
            e = energy(x, y, model)
            ys′[i] = log((1f0 - c.Δτ * e / c.N) * exp(y))
        end
        model = makemodel(xs, ys′)
        xs = [rand([1f0, -1f0], c.N) for i in 1:c.num]
        ys = [inference(model, x) for x in xs]
        outdata = (xs, ys)
        open(io -> serialize(io, outdata), "./data/" * filenames[it+1], "w")
    end
end

function measure()
    touch("./data/" * filename)
    # Imaginary roop
    for it in 1:c.iT
        xydata = open(deserialize, "./data/" * filenames[it+1])
        xs, ys = xydata
        model = makemodel(xs, ys)

        # numialize Physical Value
        energy  = 0f0im
        venergy = 0f0im
        magnet  = 0f0
        energy, venergy, magnet = sampling(model)

        # Write Data
        open("./data/" * filename, "a") do io
            write(io, string(it))
            write(io, "\t")
            write(io, string(energy))
            write(io, "\t")
            write(io, string(magnet))
            write(io, "\n")
        end
    end 
end

function sampling(model::GPmodel)
    E  = 0f0im
    vE = 0f0im
    magnet = 0f0
    # Metropolice sampling
    xs, ys = mh(model)
    
    # Calculate Physical Value
    @simd for n in 1:length(xs)
        x = xs[n]
        y = ys[n]
        e = energy(x, y, model) / c.N
        h = sum(@views x[1:c.N]) / c.N
        E  += e
        vE += (c.l - e) * conj(c.l - e)
        magnet  += h
    end
    real(E) / c.iters, real(vE) / c.iters, magnet / c.iters
end

function mh(model::GPmodel)
    outxs = Vector{Vector{Float32}}(undef, c.iters)
    outys = Vector{Complex{Float32}}(undef, c.iters)
    for i in 1:c.burnintime
        x, y = update(model)
    end
    for i in 1:c.iters
        x, y = update(model)
        outxs[i] = x
        outys[i] = y
    end
    outxs, outys
end
