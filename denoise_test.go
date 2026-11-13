package imaging

import (
	"image"
	"testing"
)

// solidImage builds an NRGBA image filled with one color.
func solidImage(w, h int, c [3]uint8) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i] = c[0]
		img.Pix[i+1] = c[1]
		img.Pix[i+2] = c[2]
		img.Pix[i+3] = 255
	}
	return img
}

func TestMedianFilterRemovesImpulse(t *testing.T) {
	// A 5x5 image at value 40 with a single salt pixel at 255. A 3x3 median
	// window must completely remove the impulse at its location.
	img := solidImage(5, 5, [3]uint8{40, 40, 40})
	img.Pix[(2*5+2)*4] = 255

	out := MedianFilter(img, &DenoiseParams{Radius: 1})
	if got := out.Pix[(2*5+2)*4]; got != 40 {
		t.Fatalf("impulse not removed: got %d, want 40", got)
	}
}

func TestMedianFilterKnownWindow(t *testing.T) {
	// Single pixel whose 3x3 R-channel neighborhood is 1..9 -> median 5.
	img := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	values := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9}
	for i, v := range values {
		img.Pix[i*4] = v
		img.Pix[i*4+1] = v
		img.Pix[i*4+2] = v
		img.Pix[i*4+3] = 255
	}
	out := MedianFilter(img, &DenoiseParams{Radius: 1})
	if got := out.Pix[(1*3+1)*4]; got != 5 {
		t.Fatalf("median of 1..9 = %d, want 5", got)
	}
}

func TestBilateralFilterFlatRegion(t *testing.T) {
	// A constant image must pass through unchanged for any parameters.
	img := solidImage(12, 12, [3]uint8{73, 90, 110})
	out := BilateralFilter(img, &DenoiseParams{
		Radius: 2, SigmaSpatial: 2.0, SigmaRange: 32.0,
	})
	for i := 0; i < len(img.Pix); i++ {
		if out.Pix[i] != img.Pix[i] {
			t.Fatalf("flat region changed at %d: got %d want %d",
				i, out.Pix[i], img.Pix[i])
		}
	}
}

func TestBilateralFilterPreservesStrongEdge(t *testing.T) {
	// Half black / half white image: the edge pixels must stay close to their
	// own side because the range weight suppresses cross-edge averaging.
	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			v := uint8(0)
			if x >= 4 {
				v = 255
			}
			pos := (y*8 + x) * 4
			img.Pix[pos], img.Pix[pos+1], img.Pix[pos+2] = v, v, v
			img.Pix[pos+3] = 255
		}
	}
	out := BilateralFilter(img, &DenoiseParams{
		Radius: 2, SigmaSpatial: 2.0, SigmaRange: 32.0,
	})
	// The dark-side pixel adjacent to the edge must remain dark (no bleed).
	dark := out.Pix[(4*8+3)*4]
	if dark > 20 {
		t.Fatalf("edge bled into dark region: value %d, want <= 20", dark)
	}
	bright := out.Pix[(4*8+4)*4]
	if bright < 235 {
		t.Fatalf("edge bled into bright region: value %d, want >= 235", bright)
	}
}

func TestDenoiseNilParamsUseNightDefaults(t *testing.T) {
	img := solidImage(6, 6, [3]uint8{50, 50, 50})
	// Nil params must not panic and must produce the same size output.
	m := MedianFilter(img, nil)
	b := BilateralFilter(img, nil)
	if m.Bounds() != img.Bounds() || b.Bounds() != img.Bounds() {
		t.Fatal("nil params produced wrong-sized output")
	}
}

func TestDenoiseDimensionsAndAlpha(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 7, 5))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2] = 100, 110, 120
		img.Pix[i+3] = 200
	}
	for _, out := range []*image.NRGBA{
		MedianFilter(img, nil),
		BilateralFilter(img, nil),
	} {
		if out.Bounds().Dx() != 7 || out.Bounds().Dy() != 5 {
			t.Fatalf("wrong dimensions: %s", out.Bounds())
		}
		for i := 3; i < len(out.Pix); i += 4 {
			if out.Pix[i] != 200 {
				t.Fatalf("alpha channel altered: got %d want 200", out.Pix[i])
			}
		}
	}
}
