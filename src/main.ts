import { get_threads, tfjs_cpu_benchmark, tfjs_wasm_benchmark, tfjs_webgl_benchmark } from './TfjsFunctions'
import { ort_wasm_benchmark, ort_webgl_benchmark } from './OrtFunctions'
import { Statistics } from './CommonFunctions'

const N_field = document.getElementById('N') as HTMLInputElement
const K_field = document.getElementById('K') as HTMLInputElement
const M_field = document.getElementById('M') as HTMLInputElement

const count_field = document.getElementById('count') as HTMLInputElement
const warmup_field = document.getElementById('warmup') as HTMLInputElement

const tfjs_cpu_count_field = document.getElementById('tfjs-cpu-count') as HTMLInputElement
const ort_cpu_count_field = document.getElementById('ort-cpu-count') as HTMLInputElement

const benchmark_results = document.getElementById('benchmark-results') as HTMLDivElement

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

function block_inputs() {
  const buttons = document.querySelectorAll('button')
  for (const button of buttons) {
    button.disabled = true
  }
  const inputs = document.querySelectorAll('input')
  for (const input of inputs) {
    input.disabled = true
  }
}

function unblock_inputs() {
  const buttons = document.querySelectorAll('button')
  for (const button of buttons) {
    button.disabled = false
  }

  const inputs = document.querySelectorAll('input')
  for (const input of inputs) {
    input.disabled = false
  }
}

function main_string_builder(params: Parameters, mean: number, interval: number): string {
  return `
  N = ${params.N}<br>
  K = ${params.K}<br>
  M = ${params.M}<br>
  count = ${params.count}<br>
  warmup = ${params.warmup}<br><br>

  Result = ${mean.toFixed(3)}±${interval.toFixed(3)}`
}

async function create_benchmark_paragraph(benchmark_fn: () => Promise<Statistics>, text_builder: (mean: number, interval: number) => string) {
  block_inputs()
  const paragraph = document.createElement('p')
  paragraph.innerHTML = 'Executing benchmark'
  paragraph.setAttribute('style', 'border:1px; border-style:solid; border-color:#FFFFF; padding: 1em;')
  benchmark_results.appendChild(paragraph)
  paragraph.scrollIntoView()

  const output = await new Promise((resolve) => {
    setTimeout(() => {
      resolve(benchmark_fn())
    }, 0)
  }) as Statistics

  paragraph.innerHTML = text_builder(output.mean, output.confidence_interval)
  paragraph.scrollIntoView()
  unblock_inputs()
}

document.getElementById('tfjs-wasm').onclick = async _ => {
  const params = get_parameters()
  const threads = tfjs_cpu_count_field.valueAsNumber
  const simd_enabled = (document.querySelector('input[name="tfjs-simd"]:checked') as HTMLInputElement).value === 'yes'

  await create_benchmark_paragraph(
    () => {
      return tfjs_wasm_benchmark(params.N, params.K, params.M, params.count, params.warmup, threads, simd_enabled)
    },
    (mean, interval) => {
      if (mean > 0) {
        return `TFJS WASM<br>threads=${threads}<br>simd=${simd_enabled}<br>` + main_string_builder(params, mean, interval)
      } else {
        return 'TFJS WASM<br>In TJFS multithreading works only with SIMD.<br>Enable SIMD!'
      }
    }
  )
}

document.getElementById('tfjs-webgl').onclick = async _ => {
  const params = get_parameters()
  await create_benchmark_paragraph(
    () => tfjs_webgl_benchmark(params.N, params.K, params.M, params.count, params.warmup),
    (mean, interval) => 'TFJS WEBGL<br>' + main_string_builder(params, mean, interval)
  )
}

document.getElementById('tfjs-cpu').onclick = async _ => {
  const params = get_parameters()

  await create_benchmark_paragraph(
    () => tfjs_cpu_benchmark(params.N, params.K, params.M, params.count, params.warmup),
    (mean, interval) => 'TFJS CPU<br>' + main_string_builder(params, mean, interval)
  )
}

document.getElementById('ort-wasm').onclick = async _ => {
  const params = get_parameters()
  const threads = ort_cpu_count_field.valueAsNumber
  const simd_enabled = (document.querySelector('input[name="ort-simd"]:checked') as HTMLInputElement).value === 'yes'

  await create_benchmark_paragraph(
    () => ort_wasm_benchmark(params.N, params.K, params.M, params.count, params.warmup, threads, simd_enabled),
    (mean, interval) => `ORT WASM<br>threads=${threads}<br>simd=${simd_enabled}<br>` + main_string_builder(params, mean, interval)
  )
}

document.getElementById('ort-webgl').onclick = async _ => {
  const params = get_parameters()

  await create_benchmark_paragraph(
    () => ort_webgl_benchmark(params.N, params.K, params.M, params.count, params.warmup),
    (mean, interval) => 'ORT WEBGL<br>' + main_string_builder(params, mean, interval)
  )
}

document.getElementById('clear-benchmarks').onclick = () => {
  benchmark_results.innerHTML = ''
}

get_threads().then((threads) => {
  tfjs_cpu_count_field.valueAsNumber = threads
  ort_cpu_count_field.valueAsNumber = navigator.hardwareConcurrency
})
