using Documenter
@testset "docstrings" begin
    # don't test docstrings on old versions to avoid failures due to changes in types
    if VERSION >= v"1.9"
        DocMeta.setdocmeta!(BlockBandedMatrices, :DocTestSetup, :(using BlockBandedMatrices); recursive=true)
        doctest(BlockBandedMatrices)
    end
end
