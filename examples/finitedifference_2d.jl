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



function _gaussseidel2(L, U, b, x=copy(b), y=copy(b), M=5)
    for _=1:M
        mul!(@view(y[1:end-1]) , U , @view(x[2:end]))
        y[end] = 0
        y .= b .- y
        x .= L\ y
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

n = 1000;
    Δt = (1/n^2)/4; @time Δ = finitedifference_2d(n);
    @time A = I - Δt*Δ  # 1m x 1m discretization
    b = randn(n^2)
    L = LowerTriangular(A)
    @time U = BandedBlockBandedMatrix(UpperTriangular(@view(A[1:end-1,2:end])), ([fill(n,n-1); n-1], [n-1; fill(n,n-1)]),
                                            (0,1), (0,1))
    x = copy(b)
    y = copy(x)
    @time _gaussseidel(L,U, b, x, y, 20) # 0.4s

    @time L̃ = sparse(BandedBlockBandedMatrix(L))
    @time Ũ = sparse(U)
    x = copy(b)
    y = copy(x)
    @time _gaussseidel2(L̃,Ũ, b, x, y, 20)

Ã = sparse(A)
@time qr(Ã);
@time Ã \b ;

A = randn(n,n)

n = 1000;
    As = [randn(2k+1,k) for k=1:n];

    @time for A in As
        qr(A)
    end
n = 1000;
    # As = [randn(2k+1,k) for k=1:n];

    @time for A in As
        lu(A)
    end

0.02*n

A*x - b



x̃ = copy(x)
ỹ = copy(y)

@view(y[1:end-1]) .= Mul(U , @view(x[2:end]))

mul!(@view(ỹ[1:end-1]) , Ũ , @view(x̃[2:end]))
y[end] = 0
ỹ[end] = 0
x = copy(b)
    y = copy(x)
    _gaussseidel(L,U, b, x, y, 1) - x_bb

x .= b .- y
ỹ .= b .- ỹ
x .= Ldiv(L, x)

x̃ .= L̃ \ ỹ


x - x_bb

x̃-x



(L̃ \ x) - (L \ x)

ỹ - y |> norm
x - x_bb

A*x - b


L̃ == L
Ũ == U

norm(A*x - b)
@time


@time copy(x)
L̃ = sparse(L)
Ũ = sparse(U)




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
