package imaging

import (
	"image"
	"math"
	"sync"
)

// srgbToLinear converts an sRGB-encoded color component value in range [0, 1]
// to a linear-light value.
func srgbToLinear(c float64) float64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

// linearToSrgb converts a linear-light color component value in range [0, 1]
// back to the sRGB encoding.
func linearToSrgb(c float64) float64 {
	if c <= 0.0031308 {
		return 12.92 * c
	}
	return 1.055*math.Pow(c, 1.0/2.4) - 0.055
}

// lut8SRGBToLinear maps 8-bit sRGB component values to linear-light values.
var lut8SRGBToLinear [256]float64

// lut16SRGBToLinear maps 16-bit sRGB component values to linear-light values.
// It is built lazily on the first 16-bit resize.
var (
	lut16SRGBToLinearOnce sync.Once
	lut16SRGBToLinearData [65536]float32
)

func lut16SRGBToLinear() *[65536]float32 {
	lut16SRGBToLinearOnce.Do(func() {
		for i := range lut16SRGBToLinearData {
			lut16SRGBToLinearData[i] = float32(srgbToLinear(float64(i) / 65535))
		}
	})
	return &lut16SRGBToLinearData
}

func init() {
	for i := range lut8SRGBToLinear {
		lut8SRGBToLinear[i] = srgbToLinear(float64(i) / 255)
	}
}

// is16bitImage reports whether the image stores 16 bits per color component.
func is16bitImage(img image.Image) bool {
	switch img.(type) {
	case *image.NRGBA64, *image.RGBA64, *image.Gray16:
		return true
	}
	return false
}

// scan16 scans the given rectangular region of a 16-bit image into dst as
// non-premultiplied 16-bit NRGBA values (4 uint16 values per pixel).
func scan16(img image.Image, x1, y1, x2, y2 int, dst []uint16) {
	switch img := img.(type) {
	case *image.NRGBA64:
		j := 0
		for y := y1; y < y2; y++ {
			i := y*img.Stride + x1*8
			for x := x1; x < x2; x++ {
				dst[j+0] = uint16(img.Pix[i+0])<<8 | uint16(img.Pix[i+1])
				dst[j+1] = uint16(img.Pix[i+2])<<8 | uint16(img.Pix[i+3])
				dst[j+2] = uint16(img.Pix[i+4])<<8 | uint16(img.Pix[i+5])
				dst[j+3] = uint16(img.Pix[i+6])<<8 | uint16(img.Pix[i+7])
				j += 4
				i += 8
			}
		}

	case *image.RGBA64:
		j := 0
		for y := y1; y < y2; y++ {
			i := y*img.Stride + x1*8
			for x := x1; x < x2; x++ {
				r := uint16(img.Pix[i+0])<<8 | uint16(img.Pix[i+1])
				g := uint16(img.Pix[i+2])<<8 | uint16(img.Pix[i+3])
				b := uint16(img.Pix[i+4])<<8 | uint16(img.Pix[i+5])
				a := uint16(img.Pix[i+6])<<8 | uint16(img.Pix[i+7])
				switch a {
				case 0:
					r, g, b = 0, 0, 0
				case 0xffff:
				default:
					r = uint16(uint32(r) * 0xffff / uint32(a))
					g = uint16(uint32(g) * 0xffff / uint32(a))
					b = uint16(uint32(b) * 0xffff / uint32(a))
				}
				dst[j+0] = r
				dst[j+1] = g
				dst[j+2] = b
				dst[j+3] = a
				j += 4
				i += 8
			}
		}

	case *image.Gray16:
		j := 0
		for y := y1; y < y2; y++ {
			i := y*img.Stride + x1*2
			for x := x1; x < x2; x++ {
				c := uint16(img.Pix[i+0])<<8 | uint16(img.Pix[i+1])
				dst[j+0] = c
				dst[j+1] = c
				dst[j+2] = c
				dst[j+3] = 0xffff
				j += 4
				i += 2
			}
		}
	}
}

// linearizeScanLine8 converts an 8-bit NRGBA scanline to linear-light
// component values with alpha in range [0, 1].
func linearizeScanLine8(src []uint8, dst []float64) {
	for i := 0; i < len(src); i += 4 {
		dst[i+0] = lut8SRGBToLinear[src[i+0]]
		dst[i+1] = lut8SRGBToLinear[src[i+1]]
		dst[i+2] = lut8SRGBToLinear[src[i+2]]
		dst[i+3] = float64(src[i+3]) * (1.0 / 255.0)
	}
}

// linearizeScanLine16 converts a 16-bit NRGBA scanline to linear-light
// component values with alpha in range [0, 1].
func linearizeScanLine16(src []uint16, dst []float64) {
	lut := lut16SRGBToLinear()
	for i := 0; i < len(src); i += 4 {
		dst[i+0] = float64(lut[src[i+0]])
		dst[i+1] = float64(lut[src[i+1]])
		dst[i+2] = float64(lut[src[i+2]])
		dst[i+3] = float64(src[i+3]) * (1.0 / 65535.0)
	}
}

// filterWeights accumulates the premultiplied linear-light color of the
// scanline pixels selected by the given filter weights.
func filterWeights(weights []indexWeight, lin []float64) (r, g, b, a float64) {
	for _, w := range weights {
		i := w.index * 4
		aw := lin[i+3] * w.weight
		r += lin[i+0] * aw
		g += lin[i+1] * aw
		b += lin[i+2] * aw
		a += aw
	}
	return
}

// putNRGBA64 stores four linear-light component values in range [0, 1] as
// sRGB-encoded 16-bit NRGBA in d.
func putNRGBA64(d []byte, r, g, b, a float64) {
	putUint16(d[0:2], clamp16(linearToSrgb(r)*65535))
	putUint16(d[2:4], clamp16(linearToSrgb(g)*65535))
	putUint16(d[4:6], clamp16(linearToSrgb(b)*65535))
	putUint16(d[6:8], clamp16(a*65535))
}

func putUint16(d []byte, v uint16) {
	d[0] = uint8(v >> 8)
	d[1] = uint8(v)
}

type indexWeight struct {
	index  int
	weight float64
}

func precomputeWeights(dstSize, srcSize int, filter ResampleFilter) [][]indexWeight {
	du := float64(srcSize) / float64(dstSize)
	scale := du
	if scale < 1.0 {
		scale = 1.0
	}
	ru := math.Ceil(scale * filter.Support)

	out := make([][]indexWeight, dstSize)
	tmp := make([]indexWeight, 0, dstSize*int(ru+2)*2)

	for v := 0; v < dstSize; v++ {
		fu := (float64(v)+0.5)*du - 0.5

		begin := int(math.Ceil(fu - ru))
		if begin < 0 {
			begin = 0
		}
		end := int(math.Floor(fu + ru))
		if end > srcSize-1 {
			end = srcSize - 1
		}

		var sum float64
		for u := begin; u <= end; u++ {
			w := filter.Kernel((float64(u) - fu) / scale)
			if w != 0 {
				sum += w
				tmp = append(tmp, indexWeight{index: u, weight: w})
			}
		}
		if sum != 0 {
			for i := range tmp {
				tmp[i].weight /= sum
			}
		}

		out[v] = tmp
		tmp = tmp[len(tmp):]
	}

	return out
}

// Resize resizes the image to the specified width and height using the specified resampling
// filter and returns the transformed image. If one of width or height is 0, the image aspect
// ratio is preserved.
//
// Resampling is performed in the linear color space: sRGB-encoded source pixels are
// linearized before filtering and converted back to sRGB on output. Color components are
// weighted by the alpha channel (premultiplied) during interpolation to avoid color
// fringes along transparent edges. 16-bit source images (NRGBA64, RGBA64, Gray16) are
// processed at 16-bit precision.
//
// Example:
//
//	dstImage := imaging.Resize(srcImage, 800, 600, imaging.Lanczos)
//
func Resize(img image.Image, width, height int, filter ResampleFilter) *image.NRGBA {
	dstW, dstH := width, height
	if dstW < 0 || dstH < 0 {
		return &image.NRGBA{}
	}
	if dstW == 0 && dstH == 0 {
		return &image.NRGBA{}
	}

	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()
	if srcW <= 0 || srcH <= 0 {
		return &image.NRGBA{}
	}

	// If new width or height is 0 then preserve aspect ratio, minimum 1px.
	if dstW == 0 {
		tmpW := float64(dstH) * float64(srcW) / float64(srcH)
		dstW = int(math.Max(1.0, math.Floor(tmpW+0.5)))
	}
	if dstH == 0 {
		tmpH := float64(dstW) * float64(srcH) / float64(srcW)
		dstH = int(math.Max(1.0, math.Floor(tmpH+0.5)))
	}

	if srcW == dstW && srcH == dstH {
		return Clone(img)
	}

	if filter.Support <= 0 {
		// Nearest-neighbor special case.
		return resizeNearest(img, dstW, dstH)
	}

	res := image.Image(img)
	if srcW != dstW {
		res = resizeHorizontal(res, dstW, filter)
	}
	if srcH != dstH {
		res = resizeVertical(res, dstH, filter)
	}
	return toNRGBA8(res)
}

// toNRGBA8 converts a resize result to 8-bit NRGBA. 16-bit intermediate
// results are converted with rounding.
func toNRGBA8(img image.Image) *image.NRGBA {
	if img, ok := img.(*image.NRGBA); ok {
		return img
	}
	src := img.(*image.NRGBA64)
	dst := image.NewNRGBA(image.Rect(0, 0, src.Rect.Dx(), src.Rect.Dy()))
	for i := 0; i < len(dst.Pix); i++ {
		v := uint32(uint16(src.Pix[i*2])<<8 | uint16(src.Pix[i*2+1]))
		dst.Pix[i] = uint8((v*0xff + 0x7fff) / 0xffff)
	}
	return dst
}

func resizeHorizontal(img image.Image, width int, filter ResampleFilter) image.Image {
	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()
	weights := precomputeWeights(width, srcW, filter)

	if is16bitImage(img) {
		dst := image.NewNRGBA64(image.Rect(0, 0, width, srcH))
		parallel(0, srcH, func(ys <-chan int) {
			scanLine := make([]uint16, srcW*4)
			lin := make([]float64, srcW*4)
			for y := range ys {
				scan16(img, 0, y, srcW, y+1, scanLine)
				linearizeScanLine16(scanLine, lin)
				j0 := y * dst.Stride
				for x := range weights {
					r, g, b, a := filterWeights(weights[x], lin)
					if a > 0 {
						aInv := 1 / a
						j := j0 + x*8
						putNRGBA64(dst.Pix[j:j+8:j+8], r*aInv, g*aInv, b*aInv, a)
					}
				}
			}
		})
		return dst
	}

	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, width, srcH))
	parallel(0, srcH, func(ys <-chan int) {
		scanLine := make([]uint8, srcW*4)
		lin := make([]float64, srcW*4)
		for y := range ys {
			src.scan(0, y, srcW, y+1, scanLine)
			linearizeScanLine8(scanLine, lin)
			j0 := y * dst.Stride
			for x := range weights {
				r, g, b, a := filterWeights(weights[x], lin)
				if a > 0 {
					aInv := 1 / a
					j := j0 + x*4
					d := dst.Pix[j : j+4 : j+4]
					d[0] = clamp(linearToSrgb(r*aInv) * 255)
					d[1] = clamp(linearToSrgb(g*aInv) * 255)
					d[2] = clamp(linearToSrgb(b*aInv) * 255)
					d[3] = clamp(a * 255)
				}
			}
		}
	})
	return dst
}

func resizeVertical(img image.Image, height int, filter ResampleFilter) image.Image {
	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()
	weights := precomputeWeights(height, srcH, filter)

	if is16bitImage(img) {
		dst := image.NewNRGBA64(image.Rect(0, 0, srcW, height))
		parallel(0, srcW, func(xs <-chan int) {
			scanLine := make([]uint16, srcH*4)
			lin := make([]float64, srcH*4)
			for x := range xs {
				scan16(img, x, 0, x+1, srcH, scanLine)
				linearizeScanLine16(scanLine, lin)
				for y := range weights {
					r, g, b, a := filterWeights(weights[y], lin)
					if a > 0 {
						aInv := 1 / a
						j := y*dst.Stride + x*8
						putNRGBA64(dst.Pix[j:j+8:j+8], r*aInv, g*aInv, b*aInv, a)
					}
				}
			}
		})
		return dst
	}

	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, srcW, height))
	parallel(0, srcW, func(xs <-chan int) {
		scanLine := make([]uint8, srcH*4)
		lin := make([]float64, srcH*4)
		for x := range xs {
			src.scan(x, 0, x+1, srcH, scanLine)
			linearizeScanLine8(scanLine, lin)
			for y := range weights {
				r, g, b, a := filterWeights(weights[y], lin)
				if a > 0 {
					aInv := 1 / a
					j := y*dst.Stride + x*4
					d := dst.Pix[j : j+4 : j+4]
					d[0] = clamp(linearToSrgb(r*aInv) * 255)
					d[1] = clamp(linearToSrgb(g*aInv) * 255)
					d[2] = clamp(linearToSrgb(b*aInv) * 255)
					d[3] = clamp(a * 255)
				}
			}
		}
	})
	return dst
}

// resizeNearest is a fast nearest-neighbor resize, no filtering.
func resizeNearest(img image.Image, width, height int) *image.NRGBA {
	dst := image.NewNRGBA(image.Rect(0, 0, width, height))
	dx := float64(img.Bounds().Dx()) / float64(width)
	dy := float64(img.Bounds().Dy()) / float64(height)

	if dx > 1 && dy > 1 {
		src := newScanner(img)
		parallel(0, height, func(ys <-chan int) {
			for y := range ys {
				srcY := int((float64(y) + 0.5) * dy)
				dstOff := y * dst.Stride
				for x := 0; x < width; x++ {
					srcX := int((float64(x) + 0.5) * dx)
					src.scan(srcX, srcY, srcX+1, srcY+1, dst.Pix[dstOff:dstOff+4])
					dstOff += 4
				}
			}
		})
	} else {
		src := toNRGBA(img)
		parallel(0, height, func(ys <-chan int) {
			for y := range ys {
				srcY := int((float64(y) + 0.5) * dy)
				srcOff0 := srcY * src.Stride
				dstOff := y * dst.Stride
				for x := 0; x < width; x++ {
					srcX := int((float64(x) + 0.5) * dx)
					srcOff := srcOff0 + srcX*4
					copy(dst.Pix[dstOff:dstOff+4], src.Pix[srcOff:srcOff+4])
					dstOff += 4
				}
			}
		})
	}

	return dst
}

// Fit scales down the image using the specified resample filter to fit the specified
// maximum width and height and returns the transformed image.
//
// Example:
//
//	dstImage := imaging.Fit(srcImage, 800, 600, imaging.Lanczos)
//
func Fit(img image.Image, width, height int, filter ResampleFilter) *image.NRGBA {
	maxW, maxH := width, height

	if maxW <= 0 || maxH <= 0 {
		return &image.NRGBA{}
	}

	srcBounds := img.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()

	if srcW <= 0 || srcH <= 0 {
		return &image.NRGBA{}
	}

	if srcW <= maxW && srcH <= maxH {
		return Clone(img)
	}

	srcAspectRatio := float64(srcW) / float64(srcH)
	maxAspectRatio := float64(maxW) / float64(maxH)

	var newW, newH int
	if srcAspectRatio > maxAspectRatio {
		newW = maxW
		newH = int(float64(newW) / srcAspectRatio)
	} else {
		newH = maxH
		newW = int(float64(newH) * srcAspectRatio)
	}

	return Resize(img, newW, newH, filter)
}

// Fill creates an image with the specified dimensions and fills it with the scaled source image.
// To achieve the correct aspect ratio without stretching, the source image will be cropped.
//
// Example:
//
//	dstImage := imaging.Fill(srcImage, 800, 600, imaging.Center, imaging.Lanczos)
//
func Fill(img image.Image, width, height int, anchor Anchor, filter ResampleFilter) *image.NRGBA {
	dstW, dstH := width, height

	if dstW <= 0 || dstH <= 0 {
		return &image.NRGBA{}
	}

	srcBounds := img.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()

	if srcW <= 0 || srcH <= 0 {
		return &image.NRGBA{}
	}

	if srcW == dstW && srcH == dstH {
		return Clone(img)
	}

	if srcW >= 100 && srcH >= 100 {
		return cropAndResize(img, dstW, dstH, anchor, filter)
	}
	return resizeAndCrop(img, dstW, dstH, anchor, filter)
}

// cropAndResize crops the image to the smallest possible size that has the required aspect ratio using
// the given anchor point, then scales it to the specified dimensions and returns the transformed image.
//
// This is generally faster than resizing first, but may result in inaccuracies when used on small source images.
func cropAndResize(img image.Image, width, height int, anchor Anchor, filter ResampleFilter) *image.NRGBA {
	dstW, dstH := width, height

	srcBounds := img.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()
	srcAspectRatio := float64(srcW) / float64(srcH)
	dstAspectRatio := float64(dstW) / float64(dstH)

	var tmp *image.NRGBA
	if srcAspectRatio < dstAspectRatio {
		cropH := float64(srcW) * float64(dstH) / float64(dstW)
		tmp = CropAnchor(img, srcW, int(math.Max(1, cropH)+0.5), anchor)
	} else {
		cropW := float64(srcH) * float64(dstW) / float64(dstH)
		tmp = CropAnchor(img, int(math.Max(1, cropW)+0.5), srcH, anchor)
	}

	return Resize(tmp, dstW, dstH, filter)
}

// resizeAndCrop resizes the image to the smallest possible size that will cover the specified dimensions,
// crops the resized image to the specified dimensions using the given anchor point and returns
// the transformed image.
func resizeAndCrop(img image.Image, width, height int, anchor Anchor, filter ResampleFilter) *image.NRGBA {
	dstW, dstH := width, height

	srcBounds := img.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()
	srcAspectRatio := float64(srcW) / float64(srcH)
	dstAspectRatio := float64(dstW) / float64(dstH)

	var tmp *image.NRGBA
	if srcAspectRatio < dstAspectRatio {
		tmp = Resize(img, dstW, 0, filter)
	} else {
		tmp = Resize(img, 0, dstH, filter)
	}

	return CropAnchor(tmp, dstW, dstH, anchor)
}

// Thumbnail scales the image up or down using the specified resample filter, crops it
// to the specified width and hight and returns the transformed image.
//
// Example:
//
//	dstImage := imaging.Thumbnail(srcImage, 100, 100, imaging.Lanczos)
//
func Thumbnail(img image.Image, width, height int, filter ResampleFilter) *image.NRGBA {
	return Fill(img, width, height, Center, filter)
}

// ResampleFilter specifies a resampling filter to be used for image resizing.
//
//	General filter recommendations:
//
//	- Lanczos
//		A high-quality resampling filter for photographic images yielding sharp results.
//
//	- CatmullRom
//		A sharp cubic filter that is faster than Lanczos filter while providing similar results.
//
//	- MitchellNetravali
//		A cubic filter that produces smoother results with less ringing artifacts than CatmullRom.
//
//	- Linear
//		Bilinear resampling filter, produces a smooth output. Faster than cubic filters.
//
//	- Box
//		Simple and fast averaging filter appropriate for downscaling.
//		When upscaling it's similar to NearestNeighbor.
//
//	- NearestNeighbor
//		Fastest resampling filter, no antialiasing.
//
type ResampleFilter struct {
	Support float64
	Kernel  func(float64) float64
}

// NearestNeighbor is a nearest-neighbor filter (no anti-aliasing).
var NearestNeighbor ResampleFilter

// Box filter (averaging pixels).
var Box ResampleFilter

// Linear filter.
var Linear ResampleFilter

// Hermite cubic spline filter (BC-spline; B=0; C=0).
var Hermite ResampleFilter

// MitchellNetravali is Mitchell-Netravali cubic filter (BC-spline; B=1/3; C=1/3).
var MitchellNetravali ResampleFilter

// CatmullRom is a Catmull-Rom - sharp cubic filter (BC-spline; B=0; C=0.5).
var CatmullRom ResampleFilter

// BSpline is a smooth cubic filter (BC-spline; B=1; C=0).
var BSpline ResampleFilter

// Gaussian is a Gaussian blurring filter.
var Gaussian ResampleFilter

// Bartlett is a Bartlett-windowed sinc filter (3 lobes).
var Bartlett ResampleFilter

// Lanczos filter (3 lobes).
var Lanczos ResampleFilter

// Hann is a Hann-windowed sinc filter (3 lobes).
var Hann ResampleFilter

// Hamming is a Hamming-windowed sinc filter (3 lobes).
var Hamming ResampleFilter

// Blackman is a Blackman-windowed sinc filter (3 lobes).
var Blackman ResampleFilter

// Welch is a Welch-windowed sinc filter (parabolic window, 3 lobes).
var Welch ResampleFilter

// Cosine is a Cosine-windowed sinc filter (3 lobes).
var Cosine ResampleFilter

func bcspline(x, b, c float64) float64 {
	var y float64
	x = math.Abs(x)
	if x < 1.0 {
		y = ((12-9*b-6*c)*x*x*x + (-18+12*b+6*c)*x*x + (6 - 2*b)) / 6
	} else if x < 2.0 {
		y = ((-b-6*c)*x*x*x + (6*b+30*c)*x*x + (-12*b-48*c)*x + (8*b + 24*c)) / 6
	}
	return y
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

func init() {
	NearestNeighbor = ResampleFilter{
		Support: 0.0, // special case - not applying the filter
	}

	Box = ResampleFilter{
		Support: 0.5,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x <= 0.5 {
				return 1.0
			}
			return 0
		},
	}

	Linear = ResampleFilter{
		Support: 1.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 1.0 {
				return 1.0 - x
			}
			return 0
		},
	}

	Hermite = ResampleFilter{
		Support: 1.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 1.0 {
				return bcspline(x, 0.0, 0.0)
			}
			return 0
		},
	}

	MitchellNetravali = ResampleFilter{
		Support: 2.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 2.0 {
				return bcspline(x, 1.0/3.0, 1.0/3.0)
			}
			return 0
		},
	}

	CatmullRom = ResampleFilter{
		Support: 2.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 2.0 {
				return bcspline(x, 0.0, 0.5)
			}
			return 0
		},
	}

	BSpline = ResampleFilter{
		Support: 2.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 2.0 {
				return bcspline(x, 1.0, 0.0)
			}
			return 0
		},
	}

	Gaussian = ResampleFilter{
		Support: 2.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 2.0 {
				return math.Exp(-2 * x * x)
			}
			return 0
		},
	}

	Bartlett = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (3.0 - x) / 3.0
			}
			return 0
		},
	}

	Lanczos = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * sinc(x/3.0)
			}
			return 0
		},
	}

	Hann = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (0.5 + 0.5*math.Cos(math.Pi*x/3.0))
			}
			return 0
		},
	}

	Hamming = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (0.54 + 0.46*math.Cos(math.Pi*x/3.0))
			}
			return 0
		},
	}

	Blackman = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (0.42 - 0.5*math.Cos(math.Pi*x/3.0+math.Pi) + 0.08*math.Cos(2.0*math.Pi*x/3.0))
			}
			return 0
		},
	}

	Welch = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (1.0 - (x * x / 9.0))
			}
			return 0
		},
	}

	Cosine = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * math.Cos((math.Pi/2.0)*(x/3.0))
			}
			return 0
		},
	}
}
