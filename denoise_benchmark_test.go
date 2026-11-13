package imaging

import (
	"image"
	"math/rand"
	"testing"
)

func benchImage(b *testing.B, w, h int) *image.NRGBA {
	b.Helper()
	rng := rand.New(rand.NewSource(1))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(img.Pix); i++ {
		img.Pix[i] = uint8(rng.Intn(256))
	}
	return img
}

func BenchmarkMedianFilter_r2(b *testing.B) {
	src := benchImage(b, 4000, 2500)
	params := DenoiseParams{Radius: 2}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MedianFilter(src, &params)
	}
}

func BenchmarkMedianFilter_nightDefault(b *testing.B) {
	src := benchImage(b, 4000, 2500)
	params := NightMedianParams
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MedianFilter(src, &params)
	}
}

func BenchmarkBilateralFilter_r2(b *testing.B) {
	src := benchImage(b, 4000, 2500)
	params := DenoiseParams{Radius: 2, SigmaSpatial: 2.0, SigmaRange: 32.0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = BilateralFilter(src, &params)
	}
}

func BenchmarkBilateralFilter_nightDefault(b *testing.B) {
	src := benchImage(b, 4000, 2500)
	params := NightBilateralParams
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = BilateralFilter(src, &params)
	}
}
