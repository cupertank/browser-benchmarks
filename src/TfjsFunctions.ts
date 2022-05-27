import * as tf from '@tensorflow/tfjs-core'
import * as tfw from '@tensorflow/tfjs-backend-wasm'
import { calculate_mean_confidence_interval, create_random_array, Statistics } from './CommonFunctions'

import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl'

const wasm_factory = tf.findBackendFactory('wasm')
tfw.setWasmPaths({
  // @ts-ignore
  'tfjs-backend-wasm.wasm': new URL('../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm', import.meta.url).toString(),
  // @ts-ignore
  'tfjs-backend-wasm-simd.wasm': new URL('../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm', import.meta.url).toString(),
  // @ts-ignore
  'tfjs-backend-wasm-threaded-simd.wasm': new URL('../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm', import.meta.url).toString()
})

export async function tfjs_wasm_benchmark(n: number, k: number, m: number, count: number, warmup: number, threads: number, simd: boolean): Promise<Statistics> {
  if (simd && threads > 1) {
    tf.env().set('WASM_HAS_SIMD_SUPPORT', true)
    tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', true)
  } else if (simd) {
    tf.env().set('WASM_HAS_SIMD_SUPPORT', true)
    tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', false)
  } else if (threads > 1) {
    console.log('In TJFS multithreading works only with SIMD.\nEnable SIMD please!')
    return {
      mean: -1,
      confidence_interval: -1
    }
  } else {
    tf.env().set('WASM_HAS_SIMD_SUPPORT', false)
    tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', false)
  }

  tf.removeBackend('wasm')
  tf.registerBackend('wasm', wasm_factory)

  tfw.setThreadsCount(threads)
  await tf.setBackend('wasm')

  console.log('TFJS WASM')
  if (threads > tfw.getThreadsCount()) {
    console.log(`Max num of threads for TFJS = ${tfw.getThreadsCount()}`)
  }
  console.log(`threads = ${tfw.getThreadsCount()}`)
  console.log(`simd = ${simd}`)
  console.log(`count = ${count}`)
  console.log(`warmup = ${warmup}`)
  console.log(`N = ${n}`)
  console.log(`K = ${k}`)
  console.log(`M = ${m}`)

  for (let i = 0; i < warmup; i++) {
    const left_array = create_random_array(n * k)
    const left = tf.tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = tf.tensor(right_array, [k, m])

    const dest = tf.matMul(left, right)

    left.dispose()
    right.dispose()
    dest.dispose()
  }

  const times = new Array<number>(count)
  for (let i = 0; i < count; i++) {
    const left_array = create_random_array(n * k)
    const left = tf.tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = tf.tensor(right_array, [k, m])

    const start = performance.now()
    const dest = tf.matMul(left, right)
    const end = performance.now()

    times[i] = end - start

    left.dispose()
    right.dispose()
    dest.dispose()
  }
  return calculate_mean_confidence_interval(times)
}

export async function tfjs_webgl_benchmark(n: number, k: number, m: number, count: number, warmup: number): Promise<Statistics> {
  await tf.setBackend('webgl')

  console.log('TFJS WEBGL')
  console.log(`count = ${count}`)
  console.log(`warmup = ${warmup}`)
  console.log(`N = ${n}`)
  console.log(`K = ${k}`)
  console.log(`M = ${m}`)

  for (let i = 0; i < warmup; i++) {
    const left_array = create_random_array(n * k)
    const left = tf.tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = tf.tensor(right_array, [k, m])

    const dest = tf.matMul(left, right)

    left.dispose()
    right.dispose()
    dest.dispose()
  }

  const times = new Array<number>(count)
  for (let i = 0; i < count; i++) {
    const left_array = create_random_array(n * k)
    const left = tf.tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = tf.tensor(right_array, [k, m])

    const start = performance.now()
    const dest = tf.matMul(left, right)
    const end = performance.now()

    times[i] = end - start

    left.dispose()
    right.dispose()
    dest.dispose()
  }

  return calculate_mean_confidence_interval(times)
}

export async function tfjs_cpu_benchmark(n: number, k: number, m: number, count: number, warmup: number): Promise<Statistics> {
  await tf.setBackend('cpu')

  console.log('TFJS CPU')
  console.log(`count = ${count}`)
  console.log(`warmup = ${warmup}`)
  console.log(`N = ${n}`)
  console.log(`K = ${k}`)
  console.log(`M = ${m}`)

  for (let i = 0; i < warmup; i++) {
    const left_array = create_random_array(n * k)
    const left = tf.tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = tf.tensor(right_array, [k, m])

    const dest = tf.matMul(left, right)

    left.dispose()
    right.dispose()
    dest.dispose()
  }

  const times = new Array<number>(count)
  for (let i = 0; i < count; i++) {
    const left_array = create_random_array(n * k)
    const left = tf.tensor(left_array, [n, k])

    const right_array = create_random_array(k * m)
    const right = tf.tensor(right_array, [k, m])

    const start = performance.now()
    const dest = tf.matMul(left, right)
    const end = performance.now()

    times[i] = end - start

    left.dispose()
    right.dispose()
    dest.dispose()
  }

  return calculate_mean_confidence_interval(times)
}

export async function get_threads(): Promise<number> {
  await tf.setBackend('wasm')
  return tfw.getThreadsCount()
}
