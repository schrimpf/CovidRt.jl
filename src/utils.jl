"""
   function lagdiff(x::AbstractVector, d=1)

Returns `out[i] = x[i] - x[i-d]` with out padded with missings so that `length(out)==length(x)`.
"""
function lagdiff(x::AbstractVector, d=1)
  if d>0
    dx = vcat(fill(missing, d), x[(d+1):end] .- x[1:(end-d)])
  elseif d<0
    dx = vcat(-x[(d+1):end] .+ x[1:(end-d)], fill(missing,d))
  else
    dx = zero(x)
  end
  return(dx)
end

"""
    function smooth(x::AbstractVector; w=pdf(Normal(), range(-3, 3, length=7)))

Returns the running weighted mean of x*w. When `n=length(w)`, then
`out[i] = sum(x[i - n÷2 + j-1] * w[j] for j in 1:n)/sum(w)`.
"""
function smooth(x::AbstractVector;  w=pdf(Normal(), range(-3, 3, length=7)))
  sx = Vector{Union{Missing, Float64}}(undef, length(x))
  n = length(w)
  if n % 2 != 1
    error("only odd length windows allowed")
  end
  shift = -(n÷2):(n÷2)
  for i in 1:length(sx)
    s = findfirst((i .+ shift) .> 0)
    l = findlast((i .+ shift) .< length(x))
    sx[i] = sum(w[s:l].*x[i .+ shift[s:l]])./sum(w[s:l])
    if isnan(sx[i])
      sx[i] = 0
    end
  end
  return(sx)
end
