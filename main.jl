include("./gpmain.jl")
using .GaussianProcessTPQ

function main()
    # Imaginary time development
    GaussianProcessTPQ.gp_imaginary_time_evolution()

    # Physical value sampling
    GaussianProcessTPQ.gp_sampling()
end

main()
