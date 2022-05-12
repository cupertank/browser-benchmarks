export function create_random_array(size: number): Float32Array {
  const array = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    array[i] = Math.random()
  }

  return array
}
