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

    for i in 1:200
        imaginarytime(model)
        ene = energy(model)
        open("./data/" * filename, "a") do io
            write(io, string(i))
            write(io, "\t")
            write(io, string(ene))
            write(io, "\n")
        end
    end
end

dirname = "./data"
rm(dirname, force=true, recursive=true)
mkdir("./data")
filename  = "physicalvalue.txt"
touch("./data/" * filename)
main(filename)
