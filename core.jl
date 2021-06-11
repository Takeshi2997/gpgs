include("./setup.jl")
include("./functions.jl")
include("./model.jl")
using Distributions, Base.Threads, Serialization, LinearAlgebra

const filenames = ["gpdata" * lpad(it, 4, "0") * ".dat" for it in 0:c.iT]
const filename  = "physicalvalue.txt"

function it_evolution(model::GPmodel)
    for it in 1:c.iT
        # Model Update!
        xs′ = [rand([1f0, -1f0], c.N) for i in 1:c.num]
        ys  = [inference(model, x) for x in xs′]
        ys′ = copy(ys)
        @simd for i in 1:c.num
            x = xs′[i]
            y = ys[i]
            e = energy(x, y, model)
            ys′[i] = log((1f0 - c.Δτ * e / c.N) * exp(y))
        end
        model = makemodel(xs′, ys′)
        outdata = (xs′, ys′)
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
        magnet  = 0f0
        energy, magnet = sampling(model)

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
        magnet  += h
    end
    real(E) / c.iters, magnet / c.iters
end

function mh(model::GPmodel)
    outxs = Vector{Vector{Float32}}(undef, c.iters)
    outys = Vector{Complex{Float32}}(undef, c.iters)
    x = rand([1f0, -1f0], c.N)
    y = inference(model, x)
    for i in 1:c.burnintime
        update(model, x, y)
    end
    @inbounds for i in 1:c.iters
        update(model, x, y)
        outxs[i] = x
        outys[i] = y
    end
    outxs, outys
end
