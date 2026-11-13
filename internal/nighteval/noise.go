package nighteval

import (
	"image"
	"math"
	"math/rand"
)

// NoiseConfig describes a night-surveillance noise mixture.
type NoiseConfig struct {
	// ReadSigma is the standard deviation of signal-independent read noise
	// (additive Gaussian), in 8-bit intensity units.
	ReadSigma float64
	// ShotScale scales signal-dependent Poisson-like shot noise. The standard
	// deviation applied to a pixel is ShotScale * sqrt(clean value).
	ShotScale float64
	// ImpulseProb is the per-pixel probability of a salt-and-pepper defect.
	ImpulseProb float64
}

// DefaultNightNoise matches a typical dark surveillance frame: dominant read
// and shot noise on dark pixels plus very sparse impulse (hot/dead pixel)
// defects. Read noise dominates, which is the regime bilateral filtering is
// designed for; impulse defects stay sparse enough that the median filter
// removes them cleanly.
var DefaultNightNoise = NoiseConfig{
	ReadSigma:   18.0,
	ShotScale:   0.45,
	ImpulseProb: 0.0008,
}

// AddNoise returns a noisy copy of src. Noise is generated independently per
// color channel. Salt-and-pepper impulses set a channel to either 0 or 255.
func AddNoise(src *image.NRGBA, cfg NoiseConfig, rng *rand.Rand) *image.NRGBA {
	w := src.Bounds().Dx()
	h := src.Bounds().Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))

	for i := 0; i < len(src.Pix); i += 4 {
		for ch := 0; ch < 3; ch++ {
			clean := float64(src.Pix[i+ch])

			shotSigma := cfg.ShotScale * math.Sqrt(clean)
			sigma := math.Hypot(cfg.ReadSigma, shotSigma)
			v := clean + rng.NormFloat64()*sigma

			if rng.Float64() < cfg.ImpulseProb {
				if rng.Intn(2) == 0 {
					v = 0
				} else {
					v = 255
				}
			}

			dst.Pix[i+ch] = clampUint8(v)
		}
		dst.Pix[i+3] = src.Pix[i+3]
	}
	return dst
}

func clampUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(math.Round(v))
}
