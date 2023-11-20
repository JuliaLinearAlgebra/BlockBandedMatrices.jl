using BlockBandedMatrices
using LinearAlgebra
using Test

using Aqua
@testset "Project quality" begin
    Aqua.test_all(BlockBandedMatrices, ambiguities=false, piracies=false)
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
