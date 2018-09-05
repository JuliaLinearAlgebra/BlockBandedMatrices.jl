###########
# This gives an example of Gauss–Seidel
#
###########

using BlockBandedMatrices, BandedMatrices, LazyArrays, FillArrays


function finitedifference_2d(n)
    h = 1/n
    D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2
    D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))
    D_yy = BandedBlockBandedMatrix(Kron(Eye(n), D²))
    D_xx + D_yy
end

function _gaussseidel(L, U, b, x=copy(b), y=copy(b), M=5)
    for _=1:M
        @view(y[1:end-1]) .= Mul(U , @view(x[2:end]))
        y[end] = 0
        x .= b .- y
        x .= Ldiv(L, x)
    end
    x
end

function gaussseidel(A, b, M)
    n = Int(sqrt(length(b)))
    L = LowerTriangular(A)
    U = BandedBlockBandedMatrix(UpperTriangular(@view(A[1:end-1,2:end])), ([fill(n,n-1); n-1], [n-1; fill(n,n-1)]),
                                            (0,1), (0,1))
    x = copy(b)
    _gaussseidel(L,U, b, x, copy(x), M) # 1.6s
    x
end

n = 200;
Δt = (1/n^2)/4; @time Δ = finitedifference_2d(n);

@time A = I - Δt*Δ  # 16k x 16k discretization
    b = randn(n^2)
    L = LowerTriangular(A)
    @time U = BandedBlockBandedMatrix(UpperTriangular(@view(A[1:end-1,2:end])), ([fill(n,n-1); n-1], [n-1; fill(n,n-1)]),
                                            (0,1), (0,1))
    x = copy(b)
    @time _gaussseidel(L,U, b, x, copy(x), 20) # 0.23s
    norm(A*x - b)


function _gaussseidel2(L, U, b, x=copy(b), y=copy(b), M=5)
    for _=1:M
        @view(y[1:end-1]) .=  U * @view(x[2:end])
        y[end] = 0
        x .= b .- y
        x = L\ x
    end
    x
end


L̃ = sparse(L)
Ũ = sparse(U)

@time _gaussseidel(L,U, b, x, copy(x), 20);
@time _gaussseidel2(L̃,Ũ, b, x, copy(x), 20);


y = x[1:end-1]
z = similar(y)
@time z .= Mul(U,y)
y = similar(x)
@time (@view(y[1:end-1]) .= Mul(U , @view(x[2:end])))
y[end] = 0
@time x .= b .- y
@time x .= Ldiv(L, x)
A = randn(9,40_000);
    @time A*x;

A
S = sparse(A);

400^2

@time L\x


h = 1/n
    @time D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2
    @time D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))
    @time D_yy = BandedBlockBandedMatrix(Kron(Eye(n), D²))
    @time D_xx .+ D_yy
    4



using Profile
dest = Δ;
    @time BlockBandedMatrices.blockbanded_copyto!(dest, D_yy)
    view(dest, Block(1,1)) .= view(dest, Block(1,1)) .+ view(D_yy, Block(1,1))
D_yy
@time copyto!(dest, Broadcast.broadcasted(+, D_xx , D_yy))




using FFTW
A = randn(400, 400);
    @time fft(A);
dest[Block(1,1)]
