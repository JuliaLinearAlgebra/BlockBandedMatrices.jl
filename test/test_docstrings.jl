using Documenter
@testset "docstrings" begin
    DocMeta.setdocmeta!(BlockBandedMatrices, :DocTestSetup, :(using BlockBandedMatrices); recursive=true)
        doctest(BlockBandedMatrices)
end
