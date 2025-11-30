import Aqua
@testset "Project quality" begin
    Aqua.test_all(BlockBandedMatrices, ambiguities=false, piracies=false)
end
