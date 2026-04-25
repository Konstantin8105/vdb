package vdb

import (
	"fmt"
	"math"
	"slices"
)

var debug bool // = true

func normalizeVector(v []float32) []float32 {
	var norm float32
	for _, val := range v {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			panic(val)
		}
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	if debug {
		fmt.Println("b", slices.Min(v), slices.Max(v), norm)
	}
	res := make([]float32, len(v))
	for i, val := range v {
		res[i] = val / norm
	}
	if debug {
		fmt.Println("a", slices.Min(v), slices.Max(v), norm)
	}
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
	}
	return
}
