



#TODO: non-matchin g blocks
isblockbanded(A::AbstractTriangular) =
    isblockbanded(parent(A))
isbandedblockbanded(A::AbstractTriangular) =
    isbandedblockbanded(parent(A))
function blockbandwidths(A::Union{UpperTriangular,UnitUpperTriangular}) 
    P = parent(A)
    if hasmatchingblocks(P)
        (min(0,blockbandwidth(P,1)), blockbandwidth(P,2))
    else
        blockbandwidths(P)
    end
end
function blockbandwidths(A::Union{LowerTriangular,UnitLowerTriangular}) 
    P = parent(A)
    if hasmatchingblocks(P)
        (blockbandwidth(P,1), min(0,blockbandwidth(P,2)))
    else
        blockbandwidths(P)
    end
end
subblockbandwidths(A::AbstractTriangular) = subblockbandwidths(parent(A))

