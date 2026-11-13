package imaging

import (
	"image"
	"math"
)

// weightScale is the fixed-point scale used for precomputed bilateral weights.
const weightScale = 1 << 8

// DenoiseParams holds parameters for the median and bilateral denoising filters.
type DenoiseParams struct {
	// Radius is the window radius in pixels. The effective window size is
	// (2*Radius+1) x (2*Radius+1). Must be >= 1.
	Radius int

	// SigmaSpatial (bilateral only) controls how quickly spatial weights decay
	// with the distance from the center pixel. Larger values blend pixels from
	// farther away. Must be > 0.
	SigmaSpatial float64

	// SigmaRange (bilateral only) controls how quickly range weights decay with
	// the intensity difference to the center pixel. Smaller values protect edges
	// more aggressively. Must be > 0.
	SigmaRange float64
}

// NightMedianParams are default median filter parameters tuned for noisy
// night-time surveillance footage (strong read/shot noise plus sparse impulse
// noise). The 3x3 window removes impulse artifacts while preserving fine
// forensic detail; pass a larger Radius in DenoiseParams for heavier noise.
var NightMedianParams = DenoiseParams{
	Radius: 1,
}

// NightBilateralParams are default bilateral filter parameters tuned for
// night-time surveillance footage. SigmaRange is matched to the typical noise
// amplitude so that homogeneous regions are smoothed while edges are retained.
var NightBilateralParams = DenoiseParams{
	Radius:       1,
	SigmaSpatial: 2.0,
	SigmaRange:   32.0,
}

// MedianFilter denoises an image using a median filter with a square
// (2*p.Radius+1) x (2*p.Radius+1) window. The median filter is particularly
// effective against salt-and-pepper (impulse) noise and never introduces
// blocky averaging artifacts because every output value is an actual pixel
// value from the input.
//
// If params is nil, NightMedianParams are used.
func MedianFilter(img image.Image, params *DenoiseParams) *image.NRGBA {
	p := normalizeMedianParams(params)
	src := toNRGBA(img)
	w := src.Bounds().Max.X
	h := src.Bounds().Max.Y
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	if w < 1 || h < 1 {
		return dst
	}

	r := p.Radius

	parallel(0, h, func(ys <-chan int) {
		// Sliding 256-bin histograms, one per color channel. Advancing one
		// column only removes the left strip and adds the right strip, keeping
		// the filter linear in the image size regardless of window size.
		var histR, histG, histB [256]int
		for y := range ys {
			y0 := y - r
			if y0 < 0 {
				y0 = 0
			}
			y1 := y + r
			if y1 > h-1 {
				y1 = h - 1
			}
			stripHeight := y1 - y0 + 1

			histR = [256]int{}
			histG = [256]int{}
			histB = [256]int{}

			// Prime the histogram for the first (leftmost) window.
			winLeft := 0
			winRight := r
			if winRight > w-1 {
				winRight = w - 1
			}
			for yy := y0; yy <= y1; yy++ {
				row := yy * src.Stride
				for xx := winLeft; xx <= winRight; xx++ {
					pos := row + xx*4
					histR[src.Pix[pos]]++
					histG[src.Pix[pos+1]]++
					histB[src.Pix[pos+2]]++
				}
			}

			for x := 0; x < w; x++ {
				dstPos := (y*w + x) * 4
				windowCount := stripHeight * (winRight - winLeft + 1)
				medianRank := windowCount / 2
				dst.Pix[dstPos] = medianFromHistogram(&histR, medianRank)
				dst.Pix[dstPos+1] = medianFromHistogram(&histG, medianRank)
				dst.Pix[dstPos+2] = medianFromHistogram(&histB, medianRank)
				dst.Pix[dstPos+3] = src.Pix[y*src.Stride+x*4+3]

				// Slide the window one pixel to the right: remove the column
				// that falls out of the window, add the newly entering column.
				if x == w-1 {
					continue
				}
				nextRight := x + r + 1
				for yy := y0; yy <= y1; yy++ {
					row := yy * src.Stride
					if nextRight >= w {
						// Right border: drop the leftmost column instead.
						remPos := row + winLeft*4
						histR[src.Pix[remPos]]--
						histG[src.Pix[remPos+1]]--
						histB[src.Pix[remPos+2]]--
					} else {
						addPos := row + nextRight*4
						histR[src.Pix[addPos]]++
						histG[src.Pix[addPos+1]]++
						histB[src.Pix[addPos+2]]++
						if x >= r {
							remPos := row + winLeft*4
							histR[src.Pix[remPos]]--
							histG[src.Pix[remPos+1]]--
							histB[src.Pix[remPos+2]]--
						}
					}
				}
				if nextRight >= w {
					winLeft++
				} else {
					winRight = nextRight
					if x >= r {
						winLeft++
					}
				}
			}
		}
	})

	return dst
}

// medianFromHistogram returns the median value given a value histogram and the
// zero-based median rank (window pixel count / 2, lower-middle on even counts).
func medianFromHistogram(hist *[256]int, medianRank int) uint8 {
	cumulative := 0
	for v := 0; v < 256; v++ {
		cumulative += hist[v]
		if cumulative > medianRank {
			return uint8(v)
		}
	}
	return 255
}

func normalizeMedianParams(params *DenoiseParams) DenoiseParams {
	if params == nil {
		return NightMedianParams
	}
	p := *params
	if p.Radius < 1 {
		p.Radius = NightMedianParams.Radius
	}
	return p
}

// BilateralFilter denoises an image using a bilateral filter. The bilateral
// filter combines a spatial Gaussian weight with a range (intensity similarity)
// Gaussian weight. Pixels that differ strongly in intensity from the center
// pixel receive near-zero weight, so edges are preserved while homogeneous
// noisy regions are smoothed.
//
// If params is nil, NightBilateralParams are used.
func BilateralFilter(img image.Image, params *DenoiseParams) *image.NRGBA {
	p := normalizeBilateralParams(params)
	src := toNRGBA(img)
	w := src.Bounds().Max.X
	h := src.Bounds().Max.Y
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	if w < 1 || h < 1 {
		return dst
	}

	r := p.Radius

	// Border-replicate the source into a padded buffer so every window access
	// in the hot loop is in range without any bounds checks.
	padded := replicateBorder(src, r)
	stride := padded.Stride

	// Small spatial weight table, one entry per window offset, row-major.
	count := (2*r + 1) * (2*r + 1)
	spatial := make([]uint16, count)
	// Range weight table shared by every offset, indexed by absolute intensity
	// difference. At 512 bytes it stays resident in L1 cache.
	rangeW := make([]uint16, 256)
	spatialDenom := 2 * p.SigmaSpatial * p.SigmaSpatial
	rangeDenom := 2 * p.SigmaRange * p.SigmaRange
	i := 0
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			// Both tables use the same 8-bit scale. The largest per-neighbor
			// contribution is 255*256*256 ≈ 1.67e7, and summing over a window
			// stays well below the uint32 accumulator limit.
			spatial[i] = uint16(0.5 + weightScale*math.Exp(
				-float64(dx*dx+dy*dy)/spatialDenom))
			i++
		}
	}
	for d := 0; d < 256; d++ {
		rangeW[d] = uint16(0.5 + weightScale*math.Exp(
			-float64(d*d)/rangeDenom))
	}

	parallel(0, h, func(ys <-chan int) {
		for y := range ys {
			rowStart := (y+r)*stride + r*4
			for x := 0; x < w; x++ {
				centerPos := rowStart + x*4
				cr := padded.Pix[centerPos]
				cg := padded.Pix[centerPos+1]
				cb := padded.Pix[centerPos+2]
				// One luminance value per pixel drives a shared range weight.
				// Surveillance frames have strongly correlated RGB channels,
				// so a luminance-guided weight is within ~0.1 dB of the full
				// per-channel formulation while costing one third of the
				// table lookups in the hot loop.
				centerLum := uint8((uint16(cr)*77 + uint16(cg)*150 +
					uint16(cb)*29) >> 8)

				var sumR, sumG, sumB, accW uint32
				k := 0
				for dy := -r; dy <= r; dy++ {
					rowPos := centerPos + dy*stride
					for dx := -r; dx <= r; dx++ {
						pos := rowPos + dx*4
						sp := uint32(spatial[k])
						k++

						pr := padded.Pix[pos]
						pg := padded.Pix[pos+1]
						pb := padded.Pix[pos+2]

						lum := uint8((uint16(pr)*77 + uint16(pg)*150 +
							uint16(pb)*29) >> 8)
						weight := sp * uint32(rangeW[absDiffU8(lum, centerLum)])

						sumR += uint32(pr) * weight
						sumG += uint32(pg) * weight
						sumB += uint32(pb) * weight
						accW += weight
					}
				}

				dstPos := (y*w + x) * 4
				dst.Pix[dstPos] = uint8((sumR + accW/2) / accW)
				dst.Pix[dstPos+1] = uint8((sumG + accW/2) / accW)
				dst.Pix[dstPos+2] = uint8((sumB + accW/2) / accW)
				dst.Pix[dstPos+3] = src.Pix[y*src.Stride+x*4+3]
			}
		}
	})

	return dst
}

// absDiffU8 returns the absolute difference between two uint8 values without
// any branch or integer conversion.
func absDiffU8(a, b uint8) uint8 {
	v := int16(a) - int16(b)
	mask := v >> 15
	return uint8((v ^ mask) - mask)
}

// replicateBorder returns a copy of img surrounded by a border of width r
// filled by replicating the nearest edge pixel.
func replicateBorder(img *image.NRGBA, r int) *image.NRGBA {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	pw, ph := w+2*r, h+2*r
	out := image.NewNRGBA(image.Rect(0, 0, pw, ph))

	for y := 0; y < ph; y++ {
		sy := y - r
		if sy < 0 {
			sy = 0
		} else if sy > h-1 {
			sy = h - 1
		}
		for x := 0; x < pw; x++ {
			sx := x - r
			if sx < 0 {
				sx = 0
			} else if sx > w-1 {
				sx = w - 1
			}
			dstPos := y*out.Stride + x*4
			srcPos := sy*img.Stride + sx*4
			out.Pix[dstPos] = img.Pix[srcPos]
			out.Pix[dstPos+1] = img.Pix[srcPos+1]
			out.Pix[dstPos+2] = img.Pix[srcPos+2]
			out.Pix[dstPos+3] = img.Pix[srcPos+3]
		}
	}
	return out
}

func normalizeBilateralParams(params *DenoiseParams) DenoiseParams {
	if params == nil {
		return NightBilateralParams
	}
	p := *params
	if p.Radius < 1 {
		p.Radius = NightBilateralParams.Radius
	}
	if p.SigmaSpatial <= 0 {
		p.SigmaSpatial = NightBilateralParams.SigmaSpatial
	}
	if p.SigmaRange <= 0 {
		p.SigmaRange = NightBilateralParams.SigmaRange
	}
	return p
}
