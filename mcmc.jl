module MCMC
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Distributions, Base.Threads, Serialization, LinearAlgebra

function imaginary(dirname::String, filename1::String)
    # Initialize Traces
    traces = Vector{Func.GPcore.Trace}(undef, Const.batchsize)
    for n in 1:Const.batchsize
        xs = [rand([1f0, -1f0], Const.dim) for i in 1:Const.init]
        bimu = zeros(Float32, 2 * Const.init)
        K  = Func.GPcore.covar(xs)
        biK1 = vcat(real.(K)/2f0,  imag.(K)/2f0)
        biK2 = vcat(-imag.(K)/2f0, real.(K)/2f0)
        biK  = hcat(biK1, biK2)
        U, Δ, V = svd(K)
        invΔ = Diagonal(1f0 ./ Δ .* (Δ .> 1f-6))
        invK = V * invΔ * U'
        biys = rand(MvNormal(bimu, biK))
        ψ = biys[1:Const.init] .+ im * biys[Const.init+1:end]
        ys  = log.(ψ)
        traces[n] = Func.GPcore.Trace(xs, ys, invK)
    end

    # Imaginary roop
    for it in 1:Const.iT
        # Initialize Physical Value
        e = zeros(Complex{Float32}, Const.batchsize)
        h = zeros(Float32, Const.batchsize)
        @threads for n in 1:Const.batchsize
            println(traces[n].ys[1])
            traces[n] = traceupdate(traces[n])
            println(traces[n].ys[1])
            exit()
            e[n], h[n] = sampling(traces[n])
        end
        energy = real(sum(e)) / Const.iters / Const.batchsize
        magnet = sum(h) / Const.iters / Const.batchsize

        # Write Data
        open(filename1, "a") do io
            write(io, string(it))
            write(io, "\t")
            write(io, string(energy / Const.dim))
            write(io, "\t")
            write(io, string(magnet / Const.dim))
            write(io, "\n")
        end

        # Trace Update!
        for n in 1:Const.batchsize
            traces[n] = Func.imaginary_evolution(traces[n])
        end
    end 

    # Write Data
    out = Vector(undef, Const.batchsize)
    for n in 1:Const.batchsize
        out[n] = (traces[n].xs, traces[n].ys)
    end
    open(io -> serialize(io, out), dirname * "/gsdata.dat", "w")
end

function sampling(trace::Func.GPcore.Trace)
    energy = 0f0im
    magnet = 0f0
    # Metropolice sampling
    xs, ys = mh(trace)
    
    # Calculate Physical Value
    for n in 1:length(xs)
        x = xs[n]
        y = ys[n]
        e = Func.energy(x, y, trace)
        h = sum(@views x[1:Const.dim])
        energy += e
        magnet += h
    end
    return energy, magnet
end

function traceupdate(trace::Func.GPcore.Trace)
    initxs = Vector{Vector{Float32}}(undef, Const.init)
    initys = Vector{Complex{Float32}}(undef, Const.init)
    for i in 1:Const.burnintime
        x, y = Func.update(trace)
    end
    for i in 1:Const.init
        x, y = Func.update(trace)
        initxs[i] = x
        initys[i] = y
    end
    K  = Func.GPcore.covar(initxs)
    U, Δ, V = svd(K)
    invΔ = Diagonal(1f0 ./ Δ .* (Δ .> 1f-6))
    invK = V * invΔ * U'
    return Func.GPcore.Trace(initxs, initys, invK)
end

function mh(trace::Func.GPcore.Trace)
    outxs  = Vector{Vector{Float32}}(undef, Const.iters)
    outys  = Vector{Complex{Float32}}(undef, Const.iters)
    for i in 1:Const.burnintime
        x, y = Func.update(trace)
    end
    for i in 1:Const.iters
        x, y = Func.update(trace)
        outxs[i] = x
        outys[i] = y
    end
    return outxs, outys
end
end
