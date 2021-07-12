include("./setup.jl")
include("./model.jl")
include("./core.jl")
using LinearAlgebra, Random, Distributions

EngArray = Vector{MersenneTwister}(undef, nthreads())
function main(filename::String)
    for i in 1:nthreads()
        EngArray[i] = MersenneTwister(i)
    end
    eng = EngArray[1]
    data_x = Vector{State}(undef, c.NData)
    data_y = Vector{Float64}(undef, c.NData)
    for i in 1:c.NData
        data_x[i] = State(rand([1.0, -1.0], c.NSpin))
    end
    bimu = zeros(Float64, 2 * c.NData)
    biI  = Array(Diagonal(ones(Float64, 2 * c.NData)))
    biψ  = rand(MvNormal(bimu, biI))
    ψ = biψ[1:c.NData] .+ im * biψ[c.NData+1:end]
    data_y = log.(ψ)
    model = GPmodel(data_x, data_y)

    batch_x = Vector{State}(undef, c.NMC)
    for i in 1:c.NMC
        x = rand(eng, [1.0, -1.0], c.NSpin)
        batch_x[i] = State(x)
    end

    logvene = 0.0
    for k in 1:200
        model = imaginarytime(model)
        ene, vene = energy(batch_x, model)
        lene = ifelse(c.l > ene, c.l - ene, 1e-6)
        entropy = logvene / c.NSpin - 2.0 * k / c.NSpin * log(lene)
        open("./data/" * filename, "a") do io
            write(io, string(k))
            write(io, "\t")
            write(io, string(ene))
            write(io, "\t")
            write(io, string(entropy))
            write(io, "\n")
        end
        logvene += log(vene)
    end
end

dirname = "./data"
rm(dirname, force=true, recursive=true)
mkdir("./data")
filename  = "physicalvalue.txt"
touch("./data/" * filename)
main(filename)
