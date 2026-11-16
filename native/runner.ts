import { dlopen, FFIType, ptr, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { join } from "node:path";

const DEFAULT_LIB_PATH = join(import.meta.dir, `libgoshawk.${suffix}`);

export interface GshkHeaderInfo {
  version: number;
  numTensors: number;
}

export class NativeGoshawkKernel {
  private lib: ReturnType<typeof dlopen>;

  constructor(libPath: string = DEFAULT_LIB_PATH) {
    if (!existsSync(libPath)) {
      throw new Error(`Native Goshawk library not found at ${libPath}. Run 'make' in native/.`);
    }

    this.lib = dlopen(libPath, {
      goshawk_rmsnorm: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.f32],
        returns: FFIType.void,
      },
      goshawk_gemv_f32: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32],
        returns: FFIType.void,
      },
      goshawk_gemv_q4: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32],
        returns: FFIType.void,
      },
      goshawk_swiglu: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32],
        returns: FFIType.void,
      },
      goshawk_inspect_gshk: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
        returns: FFIType.i32,
      },
    });
  }

  rmsnorm(x: Float32Array, weight: Float32Array, eps = 1e-6): Float32Array {
    if (x.length !== weight.length) {
      throw new Error(`Dimension mismatch: x=${x.length}, weight=${weight.length}`);
    }
    const out = new Float32Array(x.length);
    this.lib.symbols.goshawk_rmsnorm(ptr(x), ptr(weight), ptr(out), x.length, eps);
    return out;
  }

  gemvF32(W: Float32Array, x: Float32Array, outDim: number, inDim: number): Float32Array {
    if (W.length !== outDim * inDim || x.length !== inDim) {
      throw new Error("Invalid matrix-vector dimensions for gemvF32");
    }
    const out = new Float32Array(outDim);
    this.lib.symbols.goshawk_gemv_f32(ptr(W), ptr(x), ptr(out), outDim, inDim);
    return out;
  }

  gemvQ4(
    WPacked: Uint8Array,
    scales: Float32Array,
    x: Float32Array,
    outDim: number,
    inDim: number,
  ): Float32Array {
    const packedIn = Math.ceil(inDim / 2);
    if (WPacked.length !== outDim * packedIn || scales.length !== outDim || x.length !== inDim) {
      throw new Error("Invalid packed INT4 matrix-vector dimensions");
    }
    const out = new Float32Array(outDim);
    this.lib.symbols.goshawk_gemv_q4(ptr(WPacked), ptr(scales), ptr(x), ptr(out), outDim, inDim);
    return out;
  }

  swiglu(gate: Float32Array, up: Float32Array): Float32Array {
    if (gate.length !== up.length) {
      throw new Error("Dimension mismatch in swiglu");
    }
    const out = new Float32Array(gate.length);
    this.lib.symbols.goshawk_swiglu(ptr(gate), ptr(up), ptr(out), gate.length);
    return out;
  }

  inspectGshk(filePath: string): GshkHeaderInfo {
    const pathBuf = Buffer.from(`${filePath}\0`, "utf8");
    const verBuf = new Int32Array(1);
    const numBuf = new Int32Array(1);
    const rc = this.lib.symbols.goshawk_inspect_gshk(ptr(pathBuf), ptr(verBuf), ptr(numBuf));
    if (rc !== 0) {
      throw new Error(`Failed to inspect .gshk file '${filePath}' (error code ${rc})`);
    }
    return {
      version: verBuf[0] ?? 0,
      numTensors: numBuf[0] ?? 0,
    };
  }

  close(): void {
    this.lib.close();
  }
}
