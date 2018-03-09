var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#BandedMatrices.jl-Documentation-1",
    "page": "Home",
    "title": "BandedMatrices.jl Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Creating-block-banded-and-banded-block-banded-matrices-1",
    "page": "Home",
    "title": "Creating block-banded and banded-block-banded matrices",
    "category": "section",
    "text": "BlockBandedMatrixBandedBlockBandedMatrix"
},

{
    "location": "index.html#BlockBandedMatrices.blockbandwidths",
    "page": "Home",
    "title": "BlockBandedMatrices.blockbandwidths",
    "category": "function",
    "text": "blockbandwidths(A)\n\nReturns a tuple containing the upper and lower blockbandwidth of A.\n\n\n\n"
},

{
    "location": "index.html#BlockBandedMatrices.blockbandwidth",
    "page": "Home",
    "title": "BlockBandedMatrices.blockbandwidth",
    "category": "function",
    "text": "blockbandwidth(A,i)\n\nReturns the lower blockbandwidth (i==1) or the upper blockbandwidth (i==2).\n\n\n\n"
},

{
    "location": "index.html#Accessing-block-banded-and-banded-block-banded-matrices-1",
    "page": "Home",
    "title": "Accessing block-banded and banded-block-banded matrices",
    "category": "section",
    "text": "blockbandwidthsblockbandwidthsubblockbandwidthssubblockbandwidth"
},

{
    "location": "index.html#Implementation-1",
    "page": "Home",
    "title": "Implementation",
    "category": "section",
    "text": "A BlockBandedMatrix stores the entries in a single vector, ordered by columns. For example, if A is a BlockBandedMatrix with block-bandwidths (A.l,A.u) == (1,0) and the block sizes fill(2, N) where N = 3 is the number of row and column blocks, then A has zero structure[ a_11 a_12\n  a_21 a_22\n  a_31 a_32 a_33 a_34\n  a_41 a_42 a_43 a_44  \n            a_53 a_54\n            a_63 a_64 ]and is stored in memory via A.data as a single vector by columns, containing:[a_11,a_21,a_31,a_41,a_12,a_22,a_32,a_42,a_33,a_43,a_53,a_63,a_34,a_44,a_54,a_64]The reasoning behind this storage scheme as that each block still satisfies the strided matrix interface, but we can also use BLAS and LAPACK to, for example, upper-triangularize a block column all at once.A BandedBlockBandedMatrix stores the entries as a PseudoBlockMatrix, with the number of row blocks equal to A.l + A.u + 1, and the row block sizes are all A.μ + A.λ + 1. The column block sizes of the storage is the same as the the column block sizes of the BandedBlockBandedMatrix. This is a block-wise version of the storage of BandedMatrix.For example, if A is a BandedBlockBandedMatrix with block-bandwidths (A.l,A.u) == (1,0) and subblock-bandwidths (A.λ, A.μ) == (1,0), and the block sizes fill(2, N) where N = 3 is the number of row and column blocks, then A has zero structure[ a_11\n  a_21 a_22\n  a_31      a_33\n  a_41 a_42 a_43 a_44  \n            a_53    \n            a_63 a_64 ]and is stored in memory via A.data as a PseudoBlockMatrix, which has block sizes 2 x 2, containing entries:[a_11 a_22 a_33 a_44;\n a_21 X    a_43 X   ;\n a_31 a_42 a_53 a_64;\n a_41 X    a_63 X ]where X is an entry that is not used.The reasoning behind this storage scheme as that each block still satisfies the banded matrix interface."
},

]}
