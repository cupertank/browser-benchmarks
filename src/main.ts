import {get_threads, tfjs_cpu_benchmark, tfjs_wasm_benchmark, tfjs_webgl_benchmark} from "./TfjsFunctions";
import {ort_wasm_benchmark, ort_webgl_benchmark} from "./OrtFunctions";

const N_field = document.getElementById("N") as HTMLInputElement
const K_field = document.getElementById("K") as HTMLInputElement
const M_field = document.getElementById("M") as HTMLInputElement

const count_field = document.getElementById("count") as HTMLInputElement
const warmup_field = document.getElementById("warmup") as HTMLInputElement

const tfjs_cpu_count_field = document.getElementById("tfjs-cpu-count") as HTMLInputElement;
const ort_cpu_count_field = document.getElementById("ort-cpu-count") as HTMLInputElement;

type Parameters = {
  N: number,
  K: number,
  M: number,
  count: number,
  warmup: number
}
function get_parameters(): Parameters {
  return {
    N: N_field.valueAsNumber,
    K: K_field.valueAsNumber,
    M: M_field.valueAsNumber,
    count: count_field.valueAsNumber,
    warmup: warmup_field.valueAsNumber
  }
}

document.getElementById("tfjs-wasm").onclick = async _ => {
  const params = get_parameters()
  const threads = tfjs_cpu_count_field.valueAsNumber;
  const simd_enabled = (document.querySelector('input[name="tfjs-simd"]:checked') as HTMLInputElement).value == "yes";
  const output = await tfjs_wasm_benchmark(params.N, params.K, params.M, params.count, params.warmup, threads, simd_enabled);
  if (output.avg > 0) {
    console.log(`TFJS WASM time = ${output.avg.toPrecision(3)}±${output.interval.toPrecision(3)}`)
  }
}

document.getElementById("tfjs-webgl").onclick = async _ => {
  const params = get_parameters()
  const output = await tfjs_webgl_benchmark(params.N, params.K, params.M, params.count, params.warmup)
  console.log(`TFJS WEBGL time = ${output.avg.toPrecision(3)}±${output.interval.toPrecision(3)}`)
}

document.getElementById("tfjs-cpu").onclick = async _ => {
  const params = get_parameters()
  const output = await tfjs_cpu_benchmark(params.N, params.K, params.M, params.count, params.warmup)
  console.log(`TFJS CPU time = ${output.avg.toPrecision(3)}±${output.interval.toPrecision(3)}`)
}

document.getElementById("ort-wasm").onclick = async _ => {
  const params = get_parameters()

  const threads = ort_cpu_count_field.valueAsNumber
  const simd_enabled = (document.querySelector('input[name="ort-simd"]:checked') as HTMLInputElement).value == "yes";
  const output = await ort_wasm_benchmark(params.N, params.K, params.M, params.count, params.warmup, threads, simd_enabled)
  console.log(`ORT WASM time = ${output.avg.toPrecision(3)}±${output.interval.toPrecision(3)}`)
}

document.getElementById("ort-webgl").onclick = async _ => {
  const params = get_parameters()
  const output = await ort_webgl_benchmark(params.N, params.K, params.M, params.count, params.warmup)
  console.log(`ORT WEBGL time = ${output.avg.toPrecision(3)}±${output.interval.toPrecision(3)}`)
}

get_threads().then((threads) => {
  tfjs_cpu_count_field.valueAsNumber = threads
  ort_cpu_count_field.valueAsNumber = threads
})
