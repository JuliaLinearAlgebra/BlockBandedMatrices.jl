using Documenter, BlockBandedMatrices

DocMeta.setdocmeta!(BlockBandedMatrices, :DocTestSetup, :(using BlockBandedMatrices))

makedocs(
    modules = [BlockBandedMatrices],
    sitename = "BlockBandedMatrices.jl",
    pages = Any[
        "Home" => "index.md"
    ],
    warnonly = :missing_docs,
)


deploydocs(
    repo   = "github.com/JuliaLinearAlgebra/BlockBandedMatrices.jl.git",
    )
