"""
Data and priors for estimating R using Kalman filter.
"""
struct RtModel{C,P,V,T, T0, IT,TT}
  "Δlog(Δcases)"
  dlogk::C
  "Covariates shifting mean of R[t]"
  X::T
  "Covariates shifting mean of R[0]"
  X0::T0
  priors::NamedTuple{P,V}
  MA1::Bool
  "Identifier"
  id::IT
  t::TT
end

"""
    RtModel(d::Array{Vector{T}}, p::NamedTuple{P,V})

Create RtModel with `d = Δlog(Δcases)`, prior `p`, and no covariates.
"""
function RtModel(d::Array{Vector{T}}, p::NamedTuple{P,V}) where {T, P , V}
  RtModel(d, [ones(length(ds),1) for ds in d],
          fill(ones(1), length(d)), p, true, 1:length(d),
          [1:length(di) for di in d])
end

RtModel(d, x, x0, p) = RtModel(d,x,x0,p,false, 1:length(d), [1:length(di) for di in d])

"""
    RtModel(df::AbstractDataFrame, casevar::Symbol,
            xvars::Vector{Symbol}, x0vars::Vector{Symbol},
            priors::NamedTuple; L1=1, L2=1,
            time0=r->(r[casevar].>=5), t=:date, id=:fips)

Construct `RtModel` from DataFrame `df.

# Arguments
- `df`
- `casevar` variable in `df` that contains cumulative cases
- `xvars` variables in `df` shifting mean of `R[t]`
- `x0vars` variables in `df` shifting mean of `R[0]`
- `priors`
- `L1=1` lags to use to calculate `Δcases`
- `L2=1` lags to use calculage `Δlog(Δcases)`
- `time0=r->(r[casevar].>=5)` criteria for model time 0. The defaults sets t=0 when cases are first 5 or more.
- `t` time variable in `df`
- `id` id varaible in `df`
- `minvalforlog` instead of `log(Δcases)`, we take `log(min(minvalforlog, Δcases))` to avoid log(0)

"""
function RtModel(df::DataFrames.AbstractDataFrame, casevar::Symbol,
                 xvars::Vector{Symbol}, x0vars::Vector{Symbol},
                 priors::NamedTuple; L1=1, L2=1,
                 time0=r->(r[casevar].>=5), t=:date, id=:state,
                 minvalforlog=0.1)

  sdf = copy(df[!,unique(vcat(id, t, casevar, xvars..., x0vars...))])
  sort!(sdf, (id, t))

  sdf[!,:Δcases] = by(sdf, :state, Δcases = casevar => x->lagdiff(x, L1))[!,:Δcases]./L1
  sdf[!,:ΔlogΔcases] = by(sdf, :state, ΔlogΔcases =
                          :Δcases => x->lagdiff(log.(max.(x,minvalforlog)), L2))[!,:ΔlogΔcases]./L2

  sort!(sdf, (id, t))
  ids = unique(sdf[!,id])
  dlogk = Vector{Vector{Float64}}(undef, length(ids))
  dates = Vector{Vector{eltype(sdf[!,t])}}(undef, length(ids))
  X = Vector{Matrix{Float64}}(undef, length(ids))
  X0 = Vector{Vector{Float64}}(undef, length(ids))
  for (i, d) in enumerate(ids)
    gdf = filter(x->x[id].==d, sdf)
    start = findfirst(time0(gdf) .& .!(ismissing.(gdf.ΔlogΔcases)))
    dlogk[i] = Vector{Float64}(gdf[start:end, :ΔlogΔcases])
    X[i] = Matrix{Float64}(gdf[start:end, xvars])
    X0[i] = Vector{Float64}(gdf[start, x0vars])
    dates[i] = Vector{eltype(sdf[!,t])}(gdf[start:end, t])
  end

  return(RtModel(dlogk, X, X0, priors, false, ids, dates))
end


"""
    function (m::RtModel)(param)

Loglikelihood for RtModel
"""
function (m::RtModel)(param)
  Parameters.@unpack dlogk, X, X0, priors = m
  Parameters.@unpack σR, σk, σR0, γ, ρ, α, α0 = param
  logp = try
    logp = logpdf(priors.σR, σR) +
      logpdf(priors.γ, γ) +
      logpdf(priors.σR0, σR0) +
      #logpdf(priors.μR0, μR0) +
      logpdf(priors.σk, σk) +
      logpdf(priors.α, α) +
      logpdf(priors.α0, α0) +
      logpdf(priors.ρ,ρ)

    # Note that this not the usual Kalman filter because the noise in
    # dlogk is an MA(1) instead of independent. We use
    # https://arxiv.org/pdf/1909.10582.pdf
    # and adopt their notation
    A = m.MA1 ? SMatrix{2,2}([2*σk^2  -σk^2; -σk^2  2*σk^2]) : SMatrix{2,2}([2*σk^2 0; 0 2*σk^2])
    S = length(dlogk)
    T = maximum(length.(dlogk))
    meanR = Vector{typeof(σR)}(undef, T)
    zhat = Vector{typeof(σR)}(undef, T)
    varR, varRprior, K, varz, zcoef = ma1kalman_variance(T, σR0^2, ρ, σR^2, γ, A)
    for s in 1:S
      z = dlogk[s] .+ γ .- γ* (X[s]*α)
      μR0 = dot(X0[s],α0)
      ma1kalman_mean!(meanR, zhat, z, ρ, γ, μR0, K, zcoef) #, X[s]*α)
      for t in 1:length(dlogk[s])
        μ = zhat[t]
        σ = sqrt(varz[t])
        logp += logpdf(Normal(μ, σ), z[t])
      end
    end
    logp
  catch err
    @warn "Error while evaluating log likelihood. Returning -Inf.\n"*
    "The error message was:\n"*
    err.msg

    logp = -Inf*σR
  end
  return(logp)
end

"""
     mcmc(mdl::RtModel;
              θ0=(γ = 1/7, σR0=1.0, α0 = zeros(length(mdl.X0[1])),
                  σR=0.25, σk=2.0, ρ=1.0, α=zeros(size(mdl.X[1],2))),
              iterations = 2000,
              warmup=default_warmup_stages(local_optimization=nothing,
                                           stepsize_search=nothing,
                                           init_steps=100, middle_steps=100,
                                           terminating_steps=2*100,
                                           doubling_stages=3, M=Symmetric))

Estimate `RtModel` using `DynamicHMC`
"""
function mcmc(mdl::RtModel;
              θ0=(γ = 1/7, σR0=1.0, α0 = zeros(length(mdl.X0[1])),
                  σR=0.25, σk=2.0, ρ=1.0, α=zeros(size(mdl.X[1],2))),
              iterations = 2000,
              warmup=default_warmup_stages(local_optimization=nothing,
                                           stepsize_search=nothing,
                                           init_steps=100, middle_steps=100,
                                           terminating_steps=2*100,
                                           doubling_stages=3, M=Symmetric))

  K = size(mdl.X[1], 2)
  rlo = minimum(mdl.priors.ρ)
  rhi = maximum(mdl.priors.ρ)
  trans = as( (γ = asℝ₊, σR0 = asℝ₊, α0 = as(Array, length(mdl.X0[1])),
               σR = asℝ₊, σk = asℝ₊, ρ=as(Real, rlo, rhi),
               α = as(Array, K)) )
  P = TransformedLogDensity(trans, mdl)
  ∇P = ADgradient(:ForwardDiff, P)
  x0 = inverse(trans,θ0)

  rng = Random.MersenneTwister()
  res = DynamicHMC.mcmc_keep_warmup(rng, ∇P, 2000;initialization = (q = x0, ϵ=0.1),
                                    reporter = LogProgressReport(nothing, 25, 15),
                                    warmup_stages =warmup);
  post = transform.(trans,res.inference.chain)
  return(post)

end

"""
      MCMCChain(post::Array{NamedTuple,1}, xvars::Vector{Symbol}, x0vars::Vector{Symbol})

   Convert posterior from RtModel.mcmc to an MCMCChain.Chains
"""
function MCMCChain(post::Array{NamedTuple{N,V},1}, xvars::Vector{Symbol}, x0vars::Vector{Symbol}) where {N, V}
  p = post[1]
  vals = hcat([vcat([length(v)==1 ? v : vec(v) for v in values(p)]...) for p in post]...)'
  vals = reshape(vals, size(vals)..., 1)
  names = vcat([length(p[s])==1 ? String(s) : String.(s).*"[".*string.(1:length(p[s])).*"]"
                for s in keys(p)]...)
  for (i,m) in enumerate(match.(r"α0\[(\d+)\]", names))
    if (m !=nothing)
      j = parse(Int, m.captures[1])
      names[i] = "α0[$(x0vars[j])]"
    end
  end
  for (i,m) in enumerate(match.(r"α\[(\d+)\]", names))
    if (m !=nothing)
      j = parse(Int, m.captures[1])
      names[i] = "α[$(xvars[j])]"
    end
  end
  names .= replace.(names, r"\[" => s"(")
  names .= replace.(names, r"\]" => s")")
  names .= replace.(names, r"_" => s" ")
  names .= replace.(names, r"\." => s" ")
  return(MCMCChains.Chains(vals, names))
end

"""
    function plotpostr(dates, dlogk, post, X, X0; Δ=1)

Plots estimated posterior of `R[t]` on `dates` given posterior draws
of the parameters `post`, and covariates `X` and `X0`.

`post`: should be an array of named tuples of parameters, as returned
by `DynamicHMC.mcmc_with_warmup`
"""
function plotpostr(dates, dlogk, post, X, X0; Δ=1)
  k = [kalman(dlogk, p.σR, p.σk, p.σR0, dot(X0,p.α0), p.γ, p.ρ, X*p.α) for p in post];
  γ = [p.γ for p in post];
  Xa = hcat([X*p.α for p in post]...)
  meanR = hcat([x[1] for x in k]...)./Δ;
  varR = hcat([x[2] for x in k]...)./Δ^2;
  zhat = hcat([x[3] for x in k]...);
  c = "black"
  figr = plot(dates, mean(meanR, dims=2), ribbon=1.64*mean(sqrt.(varR),dims=2),
              color=c, fillalpha=0.2, linewidth=1.5, legend=:none, ylab="Rₜ")
  r=([quantile(meanR[t,:] - 1.64*sqrt.(varR[t,:]), 0.05) for t in 1:size(meanR,1)],
     [quantile(meanR[t,:] + 1.64*sqrt.(varR[t,:]), 0.95) for t in 1:size(meanR,1)])
  figr = plot!(figr, dates, zeros(length(r[1])), ribbon=(-r[1], r[2]), color=c,
               linewidth=0, ylim=nothing, fillalpha=0.2)
  T = length(dlogk)
  rt = [mean(dlogk[t]./(Δ*γ) .+ 1) for t in 1:T]
  lo = [quantile(dlogk[t]./(Δ*γ) .+ 1, 0.05) for t in 1:T]
  hi = [quantile(dlogk[t]./(Δ*γ) .+ 1, 0.95) for t in 1:T]
  figr = scatter!(figr, dates, rt, yerror=(rt-lo, hi-rt))
  figr = hline!(figr, [1.], color="red", linewidth=1.5, linestyle=:dash)
  return(figr)
end
