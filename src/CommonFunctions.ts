export type Statistics = {
  mean: number,
  confidence_interval: number
}

export function create_random_array(size: number): Float32Array {
  const array = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    array[i] = Math.random()
  }

  return array
}

export function calculate_mean_confidence_interval(times: Array<number>): Statistics {
  const count = times.length
  const sum_time = times.reduce((sum, current) => sum + current, 0)
  const avg_time = sum_time / count

  const standard_deviation = Math.sqrt(
    times.reduce((sum, current) => sum + Math.pow(current - avg_time, 2)) / (count - 1)
  )

  const sem = standard_deviation / Math.sqrt(count)
  const confidence_interval = 1.96 * sem

  return {
    mean: avg_time,
    confidence_interval
  }
}
