package imaging

import (
	"bytes"
	"image"
	"math"
	"runtime"
	"testing"
	"time"
)

func psnrNRGBA(img1, img2 *image.NRGBA) float64 {
	b1 := img1.Bounds()
	b2 := img2.Bounds()
	if b1.Dx() != b2.Dx() || b1.Dy() != b2.Dy() {
		return 0
	}
	var sum float64
	var n int
	for y := 0; y < b1.Dy(); y++ {
		for x := 0; x < b1.Dx(); x++ {
			off1 := y*img1.Stride + x*4
			off2 := y*img2.Stride + x*4
			for c := 0; c < 3; c++ {
				d := float64(img1.Pix[off1+c]) - float64(img2.Pix[off2+c])
				sum += d * d
				n++
			}
		}
	}
	if sum == 0 {
		return math.Inf(1)
	}
	mse := sum / float64(n)
	return 10 * math.Log10(255*255/mse)
}

func TestUnsharpMask(t *testing.T) {
	// Flat image: nothing to sharpen, output must equal input.
	flat := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for i := range flat.Pix {
		flat.Pix[i] = 0x80
	}
	got := UnsharpMask(flat, &UnsharpMaskProductOptions)
	if !compareNRGBA(got, toNRGBA(flat), 0) {
		t.Error("flat image was modified by UnsharpMask")
	}

	// Zero amount or non-positive radius: exact clone.
	src := toNRGBA(testdataBranchesJPG)
	for _, opts := range []*UnsharpMaskOptions{
		{Radius: 1, Amount: 0, Threshold: 0},
		{Radius: 0, Amount: 1, Threshold: 0},
		{Radius: -1, Amount: 1, Threshold: 0},
	} {
		got := UnsharpMask(testdataBranchesJPG, opts)
		if !compareNRGBA(got, src, 0) {
			t.Errorf("UnsharpMask(%+v): expected an exact clone", *opts)
		}
	}

	// Nil options: uses the portrait preset, alpha channel preserved.
	gotNil := UnsharpMask(testdataFlowersSmallPNG, nil)
	want := UnsharpMask(testdataFlowersSmallPNG, &UnsharpMaskPortraitOptions)
	if !bytes.Equal(gotNil.Pix, want.Pix) {
		t.Error("nil options do not match the portrait preset")
	}
	for i := 3; i < len(gotNil.Pix); i += 4 {
		if gotNil.Pix[i] != toNRGBA(testdataFlowersSmallPNG).Pix[i] {
			t.Fatal("alpha channel was modified")
		}
	}
}

func TestUnsharpMaskNoHalo(t *testing.T) {
	// Vertical step edge: left half dark, right half bright.
	const lo, hi = 64, 192
	src := image.NewNRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			off := y*src.Stride + x*4
			v := uint8(lo)
			if x >= 16 {
				v = hi
			}
			src.Pix[off+0] = v
			src.Pix[off+1] = v
			src.Pix[off+2] = v
			src.Pix[off+3] = 0xff
		}
	}

	// Aggressively strong settings: overshoot would show up here if uncontrolled.
	got := UnsharpMask(src, &UnsharpMaskOptions{Radius: 2, Amount: 4, Threshold: 0})
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			off := y*got.Stride + x*4
			for c := 0; c < 3; c++ {
				v := got.Pix[off+c]
				if v < lo || v > hi {
					t.Fatalf("halo detected at (%d, %d) channel %d: value %d outside [%d, %d]",
						x, y, c, v, lo, hi)
				}
			}
		}
	}

	// On a perfect binary step edge any change would be overshoot, so the
	// halo control keeps the output identical to the input.
	if !compareNRGBA(got, toNRGBA(src), 0) {
		t.Error("binary step edge was modified; halo control should prevent any overshoot")
	}
}

func TestUnsharpMaskDeterministic(t *testing.T) {
	first := UnsharpMask(testdataBranchesJPG, &UnsharpMaskProductOptions)
	for i := 0; i < 4; i++ {
		got := UnsharpMask(testdataBranchesJPG, &UnsharpMaskProductOptions)
		if !bytes.Equal(first.Pix, got.Pix) {
			t.Fatalf("run %d differs from the first run", i+2)
		}
	}

	// Result must not depend on the number of processing goroutines.
	SetMaxProcs(1)
	single := UnsharpMask(testdataBranchesJPG, &UnsharpMaskProductOptions)
	SetMaxProcs(0)
	if !bytes.Equal(first.Pix, single.Pix) {
		t.Fatal("result depends on the concurrency level")
	}
}

func TestUnsharpMaskPSNR(t *testing.T) {
	testCases := []struct {
		name    string
		src     image.Image
		options *UnsharpMaskOptions
		minPSNR float64
	}{
		{"portrait/branches", testdataBranchesJPG, &UnsharpMaskPortraitOptions, 33},
		{"portrait/flowers", testdataFlowersSmallPNG, &UnsharpMaskPortraitOptions, 33},
		{"product/branches", testdataBranchesJPG, &UnsharpMaskProductOptions, 28},
		{"product/flowers", testdataFlowersSmallPNG, &UnsharpMaskProductOptions, 28},
	}
	for _, tc := range testCases {
		got := UnsharpMask(tc.src, tc.options)
		psnr := psnrNRGBA(got, toNRGBA(tc.src))
		t.Logf("%s: PSNR = %.2f dB", tc.name, psnr)
		if psnr < tc.minPSNR {
			t.Errorf("%s: PSNR %.2f dB is below the required %.2f dB", tc.name, psnr, tc.minPSNR)
		}
		if psnr > 55 {
			t.Errorf("%s: PSNR %.2f dB is suspiciously high; sharpening had no effect", tc.name, psnr)
		}
	}
}

func TestUnsharpMaskBatch(t *testing.T) {
	const n = 1000
	src := toNRGBA(testdataFlowersSmallPNG)

	// Warm up, then record the goroutine baseline.
	UnsharpMask(src, &UnsharpMaskProductOptions)
	runtime.GC()
	before := runtime.NumGoroutine()

	first := UnsharpMask(src, &UnsharpMaskProductOptions)
	start := time.Now()
	var last *image.NRGBA
	for i := 0; i < n; i++ {
		last = UnsharpMask(src, &UnsharpMaskProductOptions)
	}
	elapsed := time.Since(start)
	t.Logf("batch of %d images (%dx%d) took %v, %.2f ms/image",
		n, src.Bounds().Dx(), src.Bounds().Dy(), elapsed, float64(elapsed.Microseconds())/1000/n)

	if !bytes.Equal(first.Pix, last.Pix) {
		t.Fatal("first and last batch results differ")
	}

	runtime.GC()
	after := runtime.NumGoroutine()
	if after > before+2 {
		t.Fatalf("goroutine leak: before=%d after=%d", before, after)
	}
}

func TestUnsharpMaskGolden(t *testing.T) {
	testCases := []struct {
		name    string
		options *UnsharpMaskOptions
	}{
		{"out_unsharp_portrait.png", &UnsharpMaskPortraitOptions},
		{"out_unsharp_product.png", &UnsharpMaskProductOptions},
	}
	for _, tc := range testCases {
		got := UnsharpMask(testdataFlowersSmallPNG, tc.options)
		want, err := Open("testdata/" + tc.name)
		if err != nil {
			t.Fatalf("failed to open image: %v", err)
		}
		if !compareNRGBAGolden(got, toNRGBA(want)) {
			t.Errorf("resulting image differs from golden: %s", tc.name)
		}
	}
}

func BenchmarkUnsharpMask(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		UnsharpMask(testdataBranchesJPG, &UnsharpMaskProductOptions)
	}
}

func BenchmarkUnsharpMaskLarge(b *testing.B) {
	src := Resize(testdataBranchesJPG, 1920, 0, Lanczos)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		UnsharpMask(src, &UnsharpMaskProductOptions)
	}
}
