module MCMC
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Distributions, Base.Threads, Random

const X = vcat(ones(Float32, Int(Const.dim / 2)), -ones(Float32, Int(Const.dim / 2)))

function imaginary(dirname::String, filename1::String)
    # Initialize Traces
    traces = Vector{Func.GPcore.Trace}(undef, Const.batchsize)
    for n in 1:Const.batchsize
        xs = shuffle(X)
        mu = zeros(Float32, Const.init)
        K  = Func.GPcore.covar(xs)
        ys = rand(MvNormal(mu, K)) .+ im .* rand(MvNormal(mu, K))
        traces[n] = Func.GPcore.Trace(xs, ys)
    end

    # Imaginary roop
    for it in 1:Const.iT
        # Initialize Physical Value
        e = zeros(Complex{Float32}, Const.batchsize)
        h = zeros(Float32, Const.batchsize)
        @threads for n in 1:Const.batchsize
            e[n], h[n] = sampling(traces[n])
        end
        energy = real(sum(e))  / Const.iters / Const.batchsize
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
end

function sampling(trace::Func.GPcore.Trace)
    traceinit = trace
    energy = 0f0im
    magnet = 0f0
    # Metropolice sampling
    xs, ys = mh(trace)
    
    # Calculate Physical Value
    for n in 1:length(xs)
        x = xs[n]
        y = ys[n]
        e = Func.energy(x, y, traceinit)
        h = sum(@views x[1:Const.dim])
        energy += e
        magnet += h
    end
    return energy, magnet
end

function mh(trace::Func.GPcore.Trace)
    outxs = Vector{Vector{Float32}}(undef, Const.iters)
    outys = Vector{Complex{Float32}}(undef, Const.iters)
    for i in 1:Const.burnintime
        xs, ys = Func.update(trace)
    end
    for i in 1:Const.iters
        xs, ys = Func.update(trace)
        outxs[i] = xs
        outys[i] = ys
    end
    return outxs, outys
end
end
