using BlockBandedMatrices
using LinearAlgebra
using Test

using Aqua
@testset "Project quality" begin
    Aqua.test_all(BlockBandedMatrices, ambiguities=false, piracies=false)
end

using Documenter
@testset "docstrings" begin
    # don't test docstrings on old versions to avoid failures due to changes in types
    if VERSION >= v"1.9"
        DocMeta.setdocmeta!(BlockBandedMatrices, :DocTestSetup, :(using BlockBandedMatrices); recursive=true)
        doctest(BlockBandedMatrices)
    end
end

include("test_blockbanded.jl")
include("test_blockskyline.jl")
include("test_bandedblockbanded.jl")
include("test_broadcasting.jl")
include("test_linalg.jl")
include("test_misc.jl")
include("test_triblockbanded.jl")
include("test_adjtransblockbanded.jl")
include("test_blockskylineqr.jl")
