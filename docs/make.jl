using Documenter, CovidRt

makedocs(;
    modules=[CovidRt],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/schrimpf/CovidRt.jl/blob/{commit}{path}#L{line}",
    sitename="CovidRt.jl",
    authors="Paul Schrimpf <paul.schrimpf@gmail.com>",
    assets=String[],
)
