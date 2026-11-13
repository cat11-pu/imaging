// Parameter sweep used during development to pick the night defaults. It is
// kept out of the main evaluation command so the shipped tool stays simple.
package main

import (
	"fmt"
	"image"
	"math/rand"

	"github.com/disintegration/imaging"
	"github.com/disintegration/imaging/internal/nighteval"
)

func main() {
	rng := rand.New(rand.NewSource(20261113))
	scenes := map[string]func(int, int) *image.NRGBA{
		"night_static":       nighteval.NightStatic,
		"night_dynamic":      nighteval.NightDynamic,
		"lowlight_forensics": nighteval.LowlightForensics,
	}

	medianRadii := []int{1, 2}
	bilateralConfigs := []imaging.DenoiseParams{
		{Radius: 1, SigmaSpatial: 1.5, SigmaRange: 18},
		{Radius: 1, SigmaSpatial: 1.5, SigmaRange: 24},
		{Radius: 1, SigmaSpatial: 1.5, SigmaRange: 32},
		{Radius: 1, SigmaSpatial: 1.0, SigmaRange: 24},
		{Radius: 2, SigmaSpatial: 2.0, SigmaRange: 18},
		{Radius: 2, SigmaSpatial: 2.0, SigmaRange: 24},
		{Radius: 2, SigmaSpatial: 2.0, SigmaRange: 32},
		{Radius: 3, SigmaSpatial: 3.0, SigmaRange: 32},
		{Radius: 3, SigmaSpatial: 3.0, SigmaRange: 45},
		{Radius: 3, SigmaSpatial: 3.0, SigmaRange: 60},
		{Radius: 4, SigmaSpatial: 3.5, SigmaRange: 45},
		{Radius: 4, SigmaSpatial: 3.5, SigmaRange: 60},
	}

	for _, name := range []string{"night_static", "night_dynamic", "lowlight_forensics"} {
		clean := scenes[name](960, 600)
		noisy := nighteval.AddNoise(clean, nighteval.DefaultNightNoise, rng)
		inputPSNR := nighteval.PSNR(clean, noisy)

		fmt.Printf("=== %s (input PSNR %.2f dB) ===\n", name, inputPSNR)
		for _, r := range medianRadii {
			out := imaging.MedianFilter(noisy, &imaging.DenoiseParams{Radius: r})
			fmt.Printf("median r=%d  PSNR=%6.2f  gain=%+6.2f  edge=%.3f  block=%.3f\n",
				r, nighteval.PSNR(clean, out), nighteval.PSNR(clean, out)-inputPSNR,
				nighteval.EdgeGradientRetention(clean, out),
				nighteval.BlockArtifactRetention(clean, out))
		}
		for _, cfg := range bilateralConfigs {
			c := cfg
			out := imaging.BilateralFilter(noisy, &c)
			fmt.Printf("bilat r=%d ss=%.1f sr=%.0f  PSNR=%6.2f  gain=%+6.2f  edge=%.3f  block=%.3f\n",
				cfg.Radius, cfg.SigmaSpatial, cfg.SigmaRange,
				nighteval.PSNR(clean, out), nighteval.PSNR(clean, out)-inputPSNR,
				nighteval.EdgeGradientRetention(clean, out),
				nighteval.BlockArtifactRetention(clean, out))
		}
		fmt.Println()
	}
}
