module AquaTests

using BlockBandedMatrices
using Test

import Aqua
@testset "Project quality" begin
    Aqua.test_all(BlockBandedMatrices, ambiguities=false, piracies=(; broken=true,))
end

end