using Documenter, BlockBandedMatrices

makedocs(modules=[BlockBandedMatrices],
			doctest = true,
			clean = true,
			format = :html,
			sitename = "BlockBandedMatrices.jl",
			authors = "Sheehan Olver",
			pages = Any[
					"Home" => "index.md"
					]
			)


deploydocs(
    repo   = "github.com/JuliaMatrices/BlockBandedMatrices.jl.git",
    )
