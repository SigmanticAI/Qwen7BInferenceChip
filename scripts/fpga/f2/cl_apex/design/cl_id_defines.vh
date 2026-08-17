// cl_id_defines.vh — CL identification registers for cl_apex.
// Values follow the aws-fpga v2.3.3 cl_axil_reg_access example convention
// (PCIe-visible CL IDs the shell reports; fpga-describe-local-image shows
// them after load). Distinguish APEX AFIs by the bridge ID/VERSION registers
// at BAR0 0x0000/0x0004 instead of changing these.
`define CL_SH_ID0 32'hF006_1D0F
`define CL_SH_ID1 32'h1D51_FEDC
