
from query_pipeline.pipeline.stages import register
from query_pipeline.steps.discover_stage import run_discover_stage
from query_pipeline.steps.post_stage import run_post_stage
from query_pipeline.steps.verify_stage import run_verify_stage

register("discover")(run_discover_stage)
register("verify")(run_verify_stage)
register("post")(run_post_stage)
