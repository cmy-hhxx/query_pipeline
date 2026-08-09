
from query_pipeline.pipeline.stages import register
from query_pipeline.steps.answer_gate_stage import run_answer_gate_stage
from query_pipeline.steps.judge_stage import run_judge_stage
from query_pipeline.steps.post_stage import run_post_stage
from query_pipeline.steps.preclean_stage import run_preclean_stage
from query_pipeline.steps.rule_gate_stage import run_rule_gate_stage
from query_pipeline.steps.segment_stage import run_segment_stage
from query_pipeline.steps.verify_stage import run_verify_stage

register("preclean")(run_preclean_stage)
register("segment")(run_segment_stage)
register("rule_gate")(run_rule_gate_stage)
register("judge")(run_judge_stage)
register("verify")(run_verify_stage)
register("answer_gate")(run_answer_gate_stage)
register("post")(run_post_stage)
