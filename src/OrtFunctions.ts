import * as ort from 'onnxruntime-web'
import { calculate_mean_confidence_interval, create_random_array, Statistics } from './CommonFunctions'

ort.env.wasm.wasmPaths = {
  // @ts-ignore
  'ort-wasm.wasm': new URL('../node_modules/onnxruntime-web/dist/ort-wasm.wasm', import.meta.url).toString(),
  // @ts-ignore
  'ort-wasm-simd.wasm': new URL('../node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm', import.meta.url).toString(),
  // @ts-ignore
  'ort-wasm-threaded.wasm': new URL('../node_modules/onnxruntime-web/dist/ort-wasm-threaded.wasm', import.meta.url).toString(),
  // @ts-ignore
  'ort-wasm-simd-threaded.wasm': new URL('../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm', import.meta.url).toString()
}

// @ts-ignore
const model_path = new URL('../model/model.onnx', import.meta.url).toString()

export async function ort_webgl_benchmark(n: number, k: number, m: number, count: number, warmup: number): Promise<Statistics> {
  console.log('ORT WEBGL')
  console.log(`count = ${count}`)
  console.log(`warmup = ${warmup}`)

  ort.env.webgl.contextId = 'webgl2'
  ort.env.webgl.pack = true
  ort.env.webgl.textureCacheMode = 'full'

  console.log(`N = ${n}`)
  console.log(`K = ${k}`)
  console.log(`M = ${m}`)

  const session = await ort.InferenceSession.create(model_path, { executionProviders: ['webgl'] })

  for (let i = 0; i < warmup; i++) {
    const left_array = create_random_array(n * k)
    const left = new ort.Tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = new ort.Tensor(right_array, [k, m])

    const feed = {
      left,
      right
    }

    // eslint-disable-next-line no-unused-vars
    const output = await session.run(feed)
  }

  const times = new Array<number>(count)
  for (let i = 0; i < count; i++) {
    const left_array = create_random_array(n * k)
    const left = new ort.Tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = new ort.Tensor(right_array, [k, m])

    const feed = {
      left,
      right
    }

    const start = performance.now()
    // eslint-disable-next-line no-unused-vars
    const output = await session.run(feed)
    const end = performance.now()

    times[i] = end - start
  }

  return calculate_mean_confidence_interval(times)

  // const sum_time = times.reduce((sum, current) => sum + current, 0)
  // const avg_time = sum_time / count
  //
  // const standard_deviation = Math.sqrt(
  //   times.reduce((sum, current) => sum + Math.pow(current - avg_time, 2)) / (count - 1)
  // )
  // const sem = standard_deviation / Math.sqrt(count)
  // const confidence_interval = 1.96 * sem
  //
  // return {
  //   avg: avg_time,
  //   interval: confidence_interval
  // }
}

export async function ort_wasm_benchmark(n: number, k: number, m: number, count: number, warmup: number, threads: number, simd: boolean): Promise<Statistics> {
  console.log('ORT WASM')

  ort.env.wasm.numThreads = threads
  ort.env.wasm.simd = simd

  const session = await ort.InferenceSession.create(model_path, { executionProviders: ['wasm'] })

  console.log(`threads = ${ort.env.wasm.numThreads}`)
  console.log(`simd = ${ort.env.wasm.simd}`)
  console.log(`count = ${count}`)
  console.log(`warmup = ${warmup}`)
  console.log(`N = ${n}`)
  console.log(`K = ${k}`)
  console.log(`M = ${m}`)

  for (let i = 0; i < warmup; i++) {
    const left_array = create_random_array(n * k)
    const left = new ort.Tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = new ort.Tensor(right_array, [k, m])

    const feed = {
      left,
      right
    }

    // eslint-disable-next-line no-unused-vars
    const output = await session.run(feed)
  }

  const times = new Array<number>(count)

  for (let i = 0; i < count; i++) {
    const left_array = create_random_array(n * k)
    const left = new ort.Tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = new ort.Tensor(right_array, [k, m])

    const feed = {
      left,
      right
    }

    const start = performance.now()
    // eslint-disable-next-line no-unused-vars
    const output = await session.run(feed)
    const end = performance.now()

    times[i] = end - start
  }

  return calculate_mean_confidence_interval(times)

  // const sum_time = times.reduce((sum, current) => sum + current, 0)
  // const avg_time = sum_time / count
  //
  // const standard_deviation = Math.sqrt(
  //   times.reduce((sum, current) => sum + Math.pow(current - avg_time, 2)) / (count - 1)
  // )
  // const sem = standard_deviation / Math.sqrt(count)
  // const confidence_interval = 1.96 * sem
  //
  // return {
  //   avg: avg_time,
  //   interval: confidence_interval
  // }
}
