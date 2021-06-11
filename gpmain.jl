module GaussianProcessTPQ
include("./setup.jl")
include("./core.jl")
include("./model.jl")
using Distributions, LinearAlgebra, Serialization

function gp_imaginary_time_evolution()
    # rm Data File make
    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir("./data")

    # Initialize Traces
    xs = [rand([1f0, -1f0], c.N) for i in 1:c.num]
    bimu = zeros(Float32, 2 * c.num)
    biI  = Array(Diagonal(ones(Float32, 2 * c.num)))
    biys = rand(MvNormal(bimu, biI))
    ψ = biys[1:c.num] .+ im * biys[c.num+1:end]
    ys = log.(ψ)
    model = makemodel(xs, ys)
    outdata = (xs, ys)
    open(io -> serialize(io, outdata), "./data/gpdata0000.dat", "w")
 
    # GP imaginary time evolution
    it_evolution(model)
end

function gp_sampling()
    # MCMC Sampling
    measure()
end
end
