package imaging

import (
	"image"
	"math"
)

// UnsharpMaskOptions are the unsharp masking parameters.
type UnsharpMaskOptions struct {
	// Radius is the sigma of the Gaussian blur used to build the mask.
	// It controls the scale of details that get amplified. Must be positive.
	Radius float64

	// Amount controls the strength of the sharpening effect.
	// 1.0 adds 100% of the masked difference back to the image.
	Amount float64

	// Threshold is the minimum absolute difference (in 0-255 units) between
	// the original and the blurred image that gets amplified. Differences
	// below the threshold are left untouched, protecting flat areas
	// (skin, skies, backgrounds) from noise amplification.
	Threshold float64
}

// Preset unsharp masking parameter sets.
var (
	// UnsharpMaskPortraitOptions is tuned for portraits: gentle strength and
	// a threshold that keeps skin texture and sensor noise from being amplified.
	UnsharpMaskPortraitOptions = UnsharpMaskOptions{Radius: 1.0, Amount: 0.4, Threshold: 2}

	// UnsharpMaskProductOptions is tuned for product shots: stronger effect
	// with a tighter radius for crisp edges and surface detail.
	UnsharpMaskProductOptions = UnsharpMaskOptions{Radius: 1.0, Amount: 0.8, Threshold: 1}
)

// UnsharpMask sharpens the image using the unsharp masking technique:
// the difference between the image and its Gaussian-blurred copy is scaled
// by Amount and added back to the image. If a nil *UnsharpMaskOptions is
// passed, UnsharpMaskPortraitOptions is used.
//
// Overshoot control: the result of every pixel is clamped to the min/max
// range of its 3x3 neighborhood in the source image, so sharpening never
// introduces values outside the locally existing range. This prevents white
// and black halos on either side of edges.
//
// Example:
//
//	dstImage := imaging.UnsharpMask(srcImage, &imaging.UnsharpMaskProductOptions)
func UnsharpMask(img image.Image, options *UnsharpMaskOptions) *image.NRGBA {
	if options == nil {
		options = &UnsharpMaskPortraitOptions
	}

	src := toNRGBA(img)
	w := src.Bounds().Max.X
	h := src.Bounds().Max.Y

	if w < 1 || h < 1 {
		return image.NewNRGBA(image.Rect(0, 0, w, h))
	}

	if options.Amount == 0 || options.Radius <= 0 {
		return Clone(src)
	}

	blurred := Blur(src, options.Radius)
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	amount := options.Amount
	threshold := options.Threshold

	parallel(0, h, func(ys <-chan int) {
		for y := range ys {
			y0 := y - 1
			if y0 < 0 {
				y0 = 0
			}
			y1 := y + 1
			if y1 > h-1 {
				y1 = h - 1
			}
			for x := 0; x < w; x++ {
				x0 := x - 1
				if x0 < 0 {
					x0 = 0
				}
				x1 := x + 1
				if x1 > w-1 {
					x1 = w - 1
				}

				var mn, mx [3]float64
				mn[0], mn[1], mn[2] = 0xff, 0xff, 0xff
				for ny := y0; ny <= y1; ny++ {
					off := ny*src.Stride + x0*4
					for nx := x0; nx <= x1; nx++ {
						s := src.Pix[off : off+3 : off+3]
						for c := 0; c < 3; c++ {
							v := float64(s[c])
							if v < mn[c] {
								mn[c] = v
							}
							if v > mx[c] {
								mx[c] = v
							}
						}
						off += 4
					}
				}

				srcOff := y*src.Stride + x*4
				dstOff := y*dst.Stride + x*4
				s := src.Pix[srcOff : srcOff+4 : srcOff+4]
				b := blurred.Pix[srcOff : srcOff+3 : srcOff+3]
				d := dst.Pix[dstOff : dstOff+4 : dstOff+4]
				for c := 0; c < 3; c++ {
					diff := float64(s[c]) - float64(b[c])
					if math.Abs(diff) < threshold {
						diff = 0
					}
					v := float64(s[c]) + amount*diff
					if v < mn[c] {
						v = mn[c]
					} else if v > mx[c] {
						v = mx[c]
					}
					d[c] = clamp(v)
				}
				d[3] = s[3]
			}
		}
	})

	return dst
}
