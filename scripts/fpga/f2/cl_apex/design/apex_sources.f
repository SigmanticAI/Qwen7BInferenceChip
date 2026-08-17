# apex_sources.f — ordered APEX RTL compile list for the F2 CL build
# (mirrors verif/top/l3/Makefile RTL_CORE + apex_top; packages first).
# Paths are relative to the apex repo root (APEX_REPO_DIR).
rtl/apex_pkg.sv
rtl/misc/f16_arith_pkg.sv
rtl/mxe/mxe_cfg_pkg.sv
rtl/xbr/stream_skid.sv
rtl/mxe/mxe_pe.sv
rtl/mxe/mxe_array.sv
rtl/mxe/mxe_buf.sv
rtl/mxe/mxe_requant.sv
rtl/mxe/mxe_ctrl.sv
rtl/mxe/mxe_top.sv
rtl/asu/asu_exp_lut.sv
rtl/asu/asu_softmax.sv
rtl/asu/rsqrt.sv
rtl/asu/asu_rmsnorm.sv
rtl/rope/rope_pair_fx.sv
rtl/rope/rope_row.sv
rtl/top/glue/apex_layer_deq.sv
rtl/top/glue/apex_residual.sv
rtl/asu/asu_silu.sv
rtl/asu/asu_swiglu.sv
rtl/kvq/cores/cq_fp_pkg.sv
rtl/kvq/cores/cq_scale_unit.sv
rtl/kvq/cores/cq_quant_unit.sv
rtl/kvq/cores/cq_dequant_unit.sv
rtl/kvq/cores/scale_bank.sv
rtl/kvq/cores/scale_bank_store.sv
rtl/kvq/cores/cq_rne_div_pipe.sv
rtl/kvq/cores/cq_quant_pipe.sv
rtl/kvq/cores/cq_scale_pipe.sv
rtl/kvq/cores/residual_buffer.sv
rtl/kvq/cores/sram_controller.sv
rtl/kvq/cores/cq_value_path.sv
rtl/kvq/cores/cq_key_path.sv
rtl/kvq/kvq_engine.sv
rtl/top/glue/apex_kvq_bank.sv
rtl/top/glue/apex_kvq_gqa_bank.sv
rtl/top/glue/apex_wcomp_bank.sv
rtl/tip/tip_decide.sv
rtl/tip/tip_importance.sv
rtl/tip/tip_top.sv
rtl/seq/seq_walker_pkg.sv
rtl/seq/seq_walker.sv
rtl/seq/seq_walker_comp.sv
rtl/seq/seq_layer_walker.sv
rtl/seq/seq_layer_walker2.sv
rtl/csr/csr_regs.sv
rtl/seam/seam_feeder_quant.sv
rtl/seam/seam_score_dequant.sv
rtl/top/glue/apex_q78_to_fp32.sv
rtl/top/glue/apex_lane32_ser.sv
rtl/top/glue/apex_lane8_unpack.sv
rtl/top/glue/apex_gam_unpack.sv
rtl/top/glue/apex_stage_buf.sv
rtl/top/glue/apex_score_fork.sv
rtl/top/glue/apex_proj_bias.sv
rtl/top/glue/apex_scale_quant.sv
# W4-DEBUT OVERLAY (scratch tree only): the D-031 W4 ingest lane sources,
# elaborated only under +define+APEX_CL_W4=1 (apex_top W4_LANE). Same three
# files verif/top/w4/tile_rtl.f compiles for the tile suite.
rtl/mxe/w4b_fp_pkg.sv
rtl/mxe/mxe_wfeed_w4b.sv
rtl/top/glue/apex_w4_ingest.sv
rtl/top/apex_top.sv
