using BlockArrays, BlockBandedMatrices, Compat.Test
    import BlockBandedMatrices: _BandedBlockBandedMatrix, scalemul!, _scalemul!, memorylayout


l , u = 1,1
λ , μ = 1,1
N = M = 10
cols = rows = 1:N

data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))

V = view(A, Block(2), Block(2))
@test unsafe_load(Base.unsafe_convert(Ptr{Float64}, V)) == 13.0




@which Base.unsafe_convert(Ptr{Float64}, V)

C =  A*A
@test C isa BandedBlockBandedMatrix
@test Matrix(A*A) ≈ Matrix(A)*Matrix(A)
@test C.l == C.u == C.λ == C.μ == 2



A = BlockBandedMatrix{Float64}(uninitialized, (rows,cols), (l,u))
    A.data .= 1:length(A.data)

V = view(A, Block(2,2))

W = 2.0Matrix(V)^2 + 3.0Matrix(V)
C = copy(V)
BLAS.gemm!('N', 'N', 2.0, V, V, 3.0, C)
@test C == W
BLAS.gemm!('N', 'N', 2.0, V, V, 3.0, V)
@test V == W

BLAS.gemm!('N', 'N', 2.0, ones(V), V, 0.0, C)
@test 2.0*ones(V)*V == C

BLAS.gemm!('N', 'N', 2.0, V, ones(V), 0.0, C)
@test 2.0*V*ones(V) == C
