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
    xs = Vector{State}(undef, c.ndata)
    for i in 1:c.ndata
        xs[i] = State(rand([1.0, -1.0], c.nspin))
    end
    bimu = zeros(Float64, 2 * c.ndata)
    biI  = Array(Diagonal(ones(Float64, 2 * c.ndata)))
    biys = rand(MvNormal(bimu, biI))
    ψ = biys[1:c.ndata] .+ im * biys[c.ndata+1:end]
    ys = log.(ψ)
    model = GPmodel(xs, ys)
    outdata = (xs, ys)
    open(io -> serialize(io, outdata), "./data/gpdata0000.dat", "w")
 
    # GP imaginary time evolution
    imaginarytime(model)
end

function gp_sampling()
    # MCMC Sampling
    measure()
end
end
