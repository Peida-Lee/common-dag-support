using LinearAlgebra
using DelimitedFiles
using Random
using TOML

config_path, output_path, source_path = ARGS
include(source_path)
LinearAlgebra.BLAS.set_num_threads(1)
config = TOML.parsefile(config_path)
Random.seed!(config["seed"])
values = readdlm(config["csv"], ',', Float64)
counts = config["counts"]
predictors = config["predictor_columns"]
response = config["response_column"]
@assert size(values, 1) == sum(counts)
started = time()
wv = DirectWassersteinVariance(values[:, predictors], values[:, response], counts, counts/sum(counts))
scores, causes, minimum, pvalues, thresholds = WV_screening(wv; α=config["alphas"], B=config["B"])
result = Dict("status"=>"completed", "source_modified"=>false,
    "entrypoint"=>"WV_screening", "B"=>config["B"], "seed"=>config["seed"],
    "raw_pvalues"=>pvalues, "author_causes_by_level"=>[Bool.(x) for x in causes],
    "author_thresholds"=>thresholds, "exclusion_scores"=>scores,
    "author_global_minimum"=>minimum, "group_counts"=>counts,
    "predictor_columns"=>predictors, "response_column"=>response,
    "rows"=>sum(counts), "seconds"=>time()-started,
    "input_role"=>"shared standardized original rows grouped by source experiment",
    "julia_version"=>string(VERSION))
open(output_path, "w") do stream
    TOML.print(stream, result)
end
println("REFERENCE_WVM_330_COMPLETED")
