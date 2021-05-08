include("./setup.jl")
include("./mcmc.jl")
using .Const, .MCMC

function main()
    # Data File make
    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    # Make Data File
    filename1 = dirname * "/physical_quantity.txt"
    touch(filename1)
 
    # Imaginary time development
    MCMC.imaginary(dirname, filename1)
end

main()
