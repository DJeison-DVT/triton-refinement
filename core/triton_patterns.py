"""Curated Triton best practices for pre-seeding pattern memory."""

TRITON_PATTERNS = [
    # --- Core element-wise pattern ---
    {
        "op_name": "element_wise",
        "pattern": "Element-wise ops: pid=tl.program_id(0), offs=pid*BLOCK+tl.arange(0,BLOCK), mask=offs<n, tl.load(ptr+offs, mask=mask), compute, tl.store(ptr+offs, result, mask=mask)",
        "outcome": "pass",
    },
    # --- Math functions ---
    {
        "op_name": "math_functions",
        "pattern": "Built-in math: tl.sqrt, tl.exp, tl.log, tl.abs, tl.sigmoid. For tanh: tl.libdevice.tanh. For sin/cos: tl.libdevice.sin/cos. Do NOT implement math manually (e.g., no Newton's method for sqrt)",
        "outcome": "pass",
    },
    # --- Wrapper conventions ---
    {
        "op_name": "wrapper_pattern",
        "pattern": "Wrapper: pass tensors directly (NOT .data_ptr()). Allocate with torch.empty_like(input). Grid: lambda meta: (triton.cdiv(n, meta['BLOCK']),). Wrapper name MUST match PyTorch function name",
        "outcome": "pass",
    },
    # --- Device handling ---
    {
        "op_name": "device_handling",
        "pattern": "All tensors on CUDA. Use device=input.device when allocating. Call .contiguous() on inputs. Never mix CPU and CUDA tensors",
        "outcome": "pass",
    },
    # --- Pointer arithmetic ---
    {
        "op_name": "pointer_arithmetic",
        "pattern": "Triton uses pointer arithmetic not indexing. Use tl.load(ptr + offsets, mask=mask) NOT tensor[i,j]. For 2D: row_offs = pid_m*BLOCK_M + tl.arange(0,BLOCK_M), flat offset = row*stride + col",
        "outcome": "pass",
    },
    # --- Reduction pattern ---
    {
        "op_name": "reduction_ops",
        "pattern": "Reductions: tl.sum(x, axis=0), tl.max(x, axis=0), tl.min(x, axis=0). For softmax: load row, subtract max for stability, exp, divide by sum. Use tl.where for conditionals, not Python if/else",
        "outcome": "pass",
    },
    # --- Masking for safety ---
    {
        "op_name": "safe_masking",
        "pattern": "Always use mask=offs<n in tl.load and tl.store. Use other=0.0 in tl.load for masked-out elements: tl.load(ptr+offs, mask=mask, other=0.0). Prevents out-of-bounds memory access",
        "outcome": "pass",
    },
    # --- Scalar fallback ---
    {
        "op_name": "scalar_fallback",
        "pattern": "When 'other' can be scalar or tensor, check isinstance(other, torch.Tensor) in wrapper. Fall back to torch op for scalar case",
        "outcome": "pass",
    },
    # --- Matrix multiply pattern ---
    {
        "op_name": "matmul_pattern",
        "pattern": "Matmul: use tl.make_block_ptr for 2D blocks, tl.dot(a, b) for tile multiply, accumulate in float32. Loop over K dimension in BLOCK_K steps. Use boundary_check=(0,1) with block pointers",
        "outcome": "pass",
    },
    # --- Multi-dimensional indexing ---
    {
        "op_name": "multi_dim_indexing",
        "pattern": "For 2D+ tensors: compute flat offsets using strides. pid_m=tl.program_id(0), pid_n=tl.program_id(1). Offset = batch*stride_batch + row*stride_row + col. Pass strides from wrapper via tensor.stride()",
        "outcome": "pass",
    },
]
