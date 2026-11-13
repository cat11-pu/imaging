package nighteval

import (
	"image"
	"math"
)

// PSNR computes the peak signal-to-noise ratio in dB between two equally sized
// NRGBA images over the R, G and B channels. A higher value means the image is
// closer to the reference.
func PSNR(ref, cmp *image.NRGBA) float64 {
	if len(ref.Pix) != len(cmp.Pix) {
		panic("PSNR: image sizes differ")
	}
	var sumSquaredError float64
	var samples float64
	for i := 0; i < len(ref.Pix); i += 4 {
		for ch := 0; ch < 3; ch++ {
			d := float64(ref.Pix[i+ch]) - float64(cmp.Pix[i+ch])
			sumSquaredError += d * d
			samples++
		}
	}
	mse := sumSquaredError / samples
	if mse == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10((255.0*255.0)/mse)
}

// luminance extracts the Rec. 601 luminance channel into a float64 grid.
func luminance(img *image.NRGBA) []float64 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	y := make([]float64, w*h)
	for i := 0; i < len(img.Pix); i += 4 {
		idx := i / 4
		y[idx] = 0.299*float64(img.Pix[i]) +
			0.587*float64(img.Pix[i+1]) +
			0.114*float64(img.Pix[i+2])
	}
	return y
}

// sobelMagnitude returns the Sobel gradient magnitude per pixel. Border pixels
// are zero.
func sobelMagnitude(lum []float64, w, h int) []float64 {
	mag := make([]float64, w*h)
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			i := y*w + x
			gx := -lum[i-w-1] + lum[i-w+1] -
				2*lum[i-1] + 2*lum[i+1] -
				lum[i+w-1] + lum[i+w+1]
			gy := -lum[i-w-1] - 2*lum[i-w] - lum[i-w+1] +
				lum[i+w-1] + 2*lum[i+w] + lum[i+w+1]
			mag[i] = math.Hypot(gx, gy)
		}
	}
	return mag
}

// EdgeGradientRetention measures whether genuine structural edges survive
// denoising, using edge gradient strength as the criterion. Strong-edge
// positions are located on the clean reference ground truth (the top 10% of
// Sobel gradient pixels), and the returned ratio is the filtered image's mean
// gradient strength at those positions divided by the clean reference's.
// A value >= 0.85 means edges are preserved; below that the filter is visibly
// softening edges. Values above 1.0 mean the output is sharper than the
// (anti-aliased) reference, which still satisfies the "no blurred edges" rule.
func EdgeGradientRetention(ref, filtered *image.NRGBA) float64 {
	w := ref.Bounds().Dx()
	h := ref.Bounds().Dy()
	refMag := sobelMagnitude(luminance(ref), w, h)
	filteredMag := sobelMagnitude(luminance(filtered), w, h)

	threshold := percentile(refMag, 90)
	var refSum, filteredSum float64
	var count float64
	for i, m := range refMag {
		if m >= threshold && m > 0 {
			refSum += m
			filteredSum += filteredMag[i]
			count++
		}
	}
	if count == 0 {
		return 1
	}
	return filteredSum / refSum
}

// percentile returns the value below which the given percentage (0-100) of
// samples fall, using a 256-bin histogram suitable for gradient magnitudes of
// 8-bit images.
func percentile(values []float64, pct float64) float64 {
	const bins = 1024
	var histogram [bins]int
	max := 1.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	for _, v := range values {
		b := int(v / max * (bins - 1))
		histogram[b]++
	}
	target := int(float64(len(values)) * pct / 100.0)
	cumulative := 0
	for b, count := range histogram {
		cumulative += count
		if cumulative >= target {
			return float64(b) / (bins - 1) * max
		}
	}
	return max
}

// BlockArtifactRetention compares the blockiness of a filtered image against
// the clean reference so that scene-inherent periodic structures cancel out.
func BlockArtifactRetention(ref, cmp *image.NRGBA) float64 {
	refRatio := blockRatio(ref)
	if refRatio == 0 {
		return blockRatio(cmp)
	}
	return blockRatio(cmp) / refRatio
}

func blockRatio(img *image.NRGBA) float64 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	lum := luminance(img)

	var boundaryEnergy, interiorEnergy float64
	var boundaryCount, interiorCount float64

	accumulate := func(i, j int, boundary bool) {
		d := math.Abs(lum[i] - lum[j])
		if boundary {
			boundaryEnergy += d
			boundaryCount++
		} else {
			interiorEnergy += d
			interiorCount++
		}
	}

	// Horizontal differences: boundaries between column 7 and 8, 15 and 16, ...
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			i := y*w + x
			accumulate(i-1, i, (x%8) == 0)
		}
	}
	// Vertical differences: boundaries between rows.
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			i := y*w + x
			accumulate(i-w, i, (y%8) == 0)
		}
	}

	if interiorCount == 0 || boundaryCount == 0 {
		return 1
	}
	return (boundaryEnergy / boundaryCount) / (interiorEnergy / interiorCount)
}
