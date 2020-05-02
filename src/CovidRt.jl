"""
   module CovidRt

Compute time and area specific estimates of the reproductive number of Covid.
"""
module CovidRt

using DynamicHMC, TransformVariables, LogDensityProblems, Parameters, Distributions
using LinearAlgebra, Plots, StatsPlots, MCMCChains
using StaticArrays, DataFrames, Random

include("utils.jl")
export smooth, lagdiff

include("kalman.jl")
export kalman

include("rt.jl")
export RtModel, plotpostr

end # module
