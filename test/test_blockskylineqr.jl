using Revise, BlockBandedMatrices

N = 10
A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (1,1))
A.data .= randn.()
A = BlockBandedMatrix(A, (1,2))
l,u = blockbandwidths(A)
K = 1
F = qr!(view(A,Block.(K:K+l),K))

qr!(randn(5,5)).factors

qr!(randn(5,5)).T