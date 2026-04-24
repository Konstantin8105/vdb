package vdb

import (
	"fmt"
	"math"
)

func normalizeVector(v []float32) []float32 {
	// TODO simd
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	res := make([]float32, len(v))
	for i, val := range v {
		res[i] = val / norm
	}
	// TODO : infinite, NAN
	return res
}

func dotProduct(a, b []float32) (dp float32, err error) {
	// error handling
	if len(a) != len(b) {
		err = fmt.Errorf("vectors have not same length")
		return
	}
	// calculate
	for i := range a {
		dp += a[i] * b[i]
		// TODO : infinite, NAN
	}
	return
}
