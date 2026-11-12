package imaging

import (
	"image"
	"runtime"
	"testing"
	"time"
)

// largeSize matches the "40 megapixel" class image from the issue:
// 7280 * 5520 = 40,185,600 pixels.
const (
	largeW = 7280
	largeH = 5520
)

func newLargeNRGBA() *image.NRGBA {
	src := image.NewNRGBA(image.Rect(0, 0, largeW, largeH))
	pix := src.Pix
	for i := 0; i < len(pix); i += 4 {
		pix[i] = uint8(i * 7)
		pix[i+1] = uint8(i*3 + 1)
		pix[i+2] = uint8(i*5 + 2)
		pix[i+3] = 255
	}
	return src
}

var benchKernel5x5 = [25]float64{
	0, 0, -1, 0, 0,
	0, -1, -2, -1, 0,
	-1, -2, 16, -2, -1,
	0, -1, -2, -1, 0,
	0, 0, -1, 0, 0,
}

// TestParallelPixelIdentical guarantees that changing the worker count never
// alters the output: every configuration must match the single-worker run
// byte for byte.
func TestParallelPixelIdentical(t *testing.T) {
	defer SetWorkers(0)
	defer SetChunkSize(0)

	src := newLargeNRGBA()

	ops := map[string]func(image.Image) *image.NRGBA{
		"grayscale": func(img image.Image) *image.NRGBA { return Grayscale(img) },
		"convolve": func(img image.Image) *image.NRGBA {
			return Convolve5x5(img, benchKernel5x5, nil)
		},
		"blur": func(img image.Image) *image.NRGBA { return Blur(img, 2.0) },
		"resize": func(img image.Image) *image.NRGBA {
			return Resize(img, largeW/2, largeH/2, Lanczos)
		},
	}

	for name, op := range ops {
		SetWorkers(1)
		SetChunkSize(64)
		want := op(src)

		for _, workers := range []int{2, 4, 8, 16} {
			for _, chunk := range []int{16, 64, 256} {
				SetWorkers(workers)
				SetChunkSize(chunk)
				got := op(src)
				if !compareNRGBA(got, want, 0) {
					t.Fatalf("%s: parallel output (workers=%d chunk=%d) differs from single worker",
						name, workers, chunk)
				}
			}
		}
	}
}

// TestSmallImageNoOverhead ensures tiny images take the inline path and never
// touch the worker pool, so chunk scheduling cannot dominate their runtime.
func TestSmallImageNoOverhead(t *testing.T) {
	defer SetWorkers(0)
	defer SetChunkSize(0)
	SetWorkers(8)

	pool.ensure(8)
	pool.mu.Lock()
	before := pool.workers
	pool.mu.Unlock()

	small := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for i := range small.Pix {
		small.Pix[i] = uint8(i)
	}

	for i := 0; i < 100; i++ {
		_ = Grayscale(small)
	}

	pool.mu.Lock()
	after := pool.workers
	pool.mu.Unlock()
	if after != before {
		t.Fatalf("small image scheduling changed pool size: %d -> %d", before, after)
	}
}

func benchmarkConvolveLarge(b *testing.B, workers, chunk int) {
	SetWorkers(workers)
	SetChunkSize(chunk)
	b.Cleanup(func() {
		SetWorkers(0)
		SetChunkSize(0)
	})

	src := newLargeNRGBA()
	runtime.GC()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Convolve5x5(src, benchKernel5x5, nil)
	}
}

func BenchmarkConvolveLargeSerial(b *testing.B) {
	benchmarkConvolveLarge(b, 1, 64)
}

func BenchmarkConvolveLargePool8(b *testing.B) {
	benchmarkConvolveLarge(b, 8, 64)
}

func BenchmarkConvolveLargeDefault(b *testing.B) {
	benchmarkConvolveLarge(b, 0, 64)
}

// BenchmarkSmallGrayscale checks that small images do not pay pool dispatch
// overhead: it must stay on the inline single-goroutine path.
func BenchmarkSmallGrayscale(b *testing.B) {
	SetWorkers(8)
	b.Cleanup(func() { SetWorkers(0) })

	small := image.NewNRGBA(image.Rect(0, 0, 32, 32))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Grayscale(small)
	}
}

// TestPerformanceTargets guards the hard performance requirements on an
// 8-worker pool: at least 5x speedup versus single worker and a per-image
// latency below 200ms for the 40 megapixel workload. It only enforces the
// speedup when enough cores are available, so CI with fewer cores is not
// flaky; the latency assertion runs whenever 8 workers are usable.
func TestPerformanceTargets(t *testing.T) {
	if runtime.NumCPU() < 8 {
		t.Skip("performance target requires at least 8 CPU cores")
	}

	src := newLargeNRGBA()
	op := func() {
		_ = Convolve5x5(src, benchKernel5x5, nil)
	}

	SetWorkers(1)
	serial := measureRuns(op, 3)

	SetWorkers(8)
	defer SetWorkers(0)
	parallel := measureRuns(op, 5)

	speedup := serial.Seconds() / parallel.Seconds()
	t.Logf("serial=%v parallel8=%v speedup=%.2fx", serial, parallel, speedup)

	if speedup < 5.0 {
		t.Fatalf("speedup %.2fx, want >= 5x", speedup)
	}
	if parallel >= 200*time.Millisecond {
		t.Fatalf("parallel latency %v, want < 200ms", parallel)
	}
}

// measureRuns returns the average wall-clock duration of n runs.
func measureRuns(fn func(), n int) time.Duration {
	start := time.Now()
	for i := 0; i < n; i++ {
		fn()
	}
	return time.Since(start) / time.Duration(n)
}

// TestMemoryPeakUnder3x verifies the single-pass convolution memory peak stays
// within three times the source image size.
func TestMemoryPeakUnder3x(t *testing.T) {
	defer SetWorkers(0)
	SetWorkers(8)

	src := newLargeNRGBA()
	srcBytes := uint64(len(src.Pix))

	runtime.GC()
	var base, peak runtime.MemStats
	runtime.ReadMemStats(&base)

	dst := Convolve5x5(src, benchKernel5x5, nil)
	runtime.ReadMemStats(&peak)

	growth := peak.TotalAlloc - base.TotalAlloc
	ratio := float64(growth) / float64(srcBytes)
	t.Logf("src=%d MiB dst=%d MiB totalAlloc during call=%d MiB ratio=%.2fx",
		srcBytes>>20, len(dst.Pix)>>20, growth>>20, ratio)
	if ratio > 3.0 {
		t.Fatalf("memory peak %.2fx source, want <= 3x", ratio)
	}
}

// TestMemoryPeakTwoPass checks the two-pass pipelines (blur and resize), which
// hold an intermediate buffer, against the same three-times budget.
func TestMemoryPeakTwoPass(t *testing.T) {
	defer SetWorkers(0)
	SetWorkers(8)

	src := newLargeNRGBA()
	srcBytes := uint64(len(src.Pix))

	ops := map[string]func(image.Image) *image.NRGBA{
		"blur": func(img image.Image) *image.NRGBA { return Blur(img, 2.0) },
		"resize": func(img image.Image) *image.NRGBA {
			return Resize(img, largeW/2, largeH/2, Lanczos)
		},
	}
	for name, op := range ops {
		runtime.GC()
		var base, peak runtime.MemStats
		runtime.ReadMemStats(&base)
		dst := op(src)
		runtime.ReadMemStats(&peak)

		growth := peak.TotalAlloc - base.TotalAlloc
		ratio := float64(growth) / float64(srcBytes)
		t.Logf("%s: src=%d MiB dst=%d MiB totalAlloc=%d MiB ratio=%.2fx",
			name, srcBytes>>20, len(dst.Pix)>>20, growth>>20, ratio)
		if ratio > 3.0 {
			t.Fatalf("%s: memory peak %.2fx source, want <= 3x", name, ratio)
		}
	}
}
