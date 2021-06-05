include("./gpmain.jl")
using .GaussianProcessTPQ

function main()
    # rm Data File make
    dirname = "./data"
    rm(dirname, force=true, recursive=true)

    # Imaginary time development
    GaussianProcessTPQ.main()
end

main()
