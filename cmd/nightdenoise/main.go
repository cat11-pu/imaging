// Command nightdenoise evaluates median and bilateral denoising filters on
// synthetic night-surveillance scenes. It generates before/after comparison
// images, a metrics report (PSNR gain, edge gradient retention, block artifact
// ratio) and single-image timing including a 40-megapixel performance check.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/disintegration/imaging"
	"github.com/disintegration/imaging/internal/nighteval"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

const (
	sampleW = 960
	sampleH = 600

	// Acceptance thresholds from the requirements.
	minPSNRGainDB     = 6.0
	minEdgeRetention  = 0.85
	maxBlockRatio     = 1.10
	max40MPLatencyMS  = 500.0
	fortyMPPixelCount = 40_000_000
)

type sample struct {
	name  string
	clean *image.NRGBA
	noisy *image.NRGBA
}

type filterResult struct {
	name          string
	output        *image.NRGBA
	inputPSNR     float64
	outputPSNR    float64
	edgeRetention float64
	blockRatio    float64
	latencyMS     float64
}

func main() {
	outDir := "artifacts/denoise"
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fatal(err)
	}

	rng := rand.New(rand.NewSource(20261113))

	sceneBuilders := map[string]func(int, int) *image.NRGBA{
		"night_static":       nighteval.NightStatic,
		"night_dynamic":      nighteval.NightDynamic,
		"lowlight_forensics": nighteval.LowlightForensics,
	}

	samples := make([]sample, 0, 3)
	for _, name := range []string{"night_static", "night_dynamic", "lowlight_forensics"} {
		clean := sceneBuilders[name](sampleW, sampleH)
		noisy := nighteval.AddNoise(clean, nighteval.DefaultNightNoise, rng)
		samples = append(samples, sample{name: name, clean: clean, noisy: noisy})
	}

	medianParams := imaging.NightMedianParams
	bilateralParams := imaging.NightBilateralParams

	var reportRows [][]string
	allPass := true

	for _, s := range samples {
		results := runFilters(s, medianParams, bilateralParams)

		comparison := buildComparison(s, results)
		path := filepath.Join(outDir, "compare_"+s.name+".png")
		if err := savePNG(path, comparison); err != nil {
			fatal(err)
		}

		for _, r := range results {
			gain := r.outputPSNR - r.inputPSNR
			pass := gain >= minPSNRGainDB &&
				r.edgeRetention >= minEdgeRetention &&
				r.blockRatio <= maxBlockRatio
			if !pass {
				allPass = false
			}
			reportRows = append(reportRows, []string{
				s.name,
				r.name,
				fmt.Sprintf("%.2f", r.inputPSNR),
				fmt.Sprintf("%.2f", r.outputPSNR),
				fmt.Sprintf("%+.2f", gain),
				fmt.Sprintf("%.3f", r.edgeRetention),
				fmt.Sprintf("%.3f", r.blockRatio),
				fmt.Sprintf("%.1f", r.latencyMS),
				boolMark(pass),
			})
		}
	}

	// 40-megapixel latency benchmark on the static night scene.
	bigClean := scaleToPixelCount(nighteval.NightStatic(sampleW, sampleH), fortyMPPixelCount)
	bigNoisy := nighteval.AddNoise(bigClean, nighteval.DefaultNightNoise, rand.New(rand.NewSource(7)))
	median40 := benchmarkFilter("median", bigNoisy, func(img image.Image) *image.NRGBA {
		return imaging.MedianFilter(img, &medianParams)
	})
	bilateral40 := benchmarkFilter("bilateral", bigNoisy, func(img image.Image) *image.NRGBA {
		return imaging.BilateralFilter(img, &bilateralParams)
	})

	if err := writeReport(outDir, reportRows, median40, bilateral40, allPass,
		medianParams, bilateralParams); err != nil {
		fatal(err)
	}

	fmt.Println("Comparison images and report written to", outDir)
	fmt.Printf("40MP latency: median %.1f ms, bilateral %.1f ms (budget %.0f ms)\n",
		median40, bilateral40, max40MPLatencyMS)
	if !allPass || median40 > max40MPLatencyMS || bilateral40 > max40MPLatencyMS {
		fmt.Println("RESULT: FAIL - one or more acceptance criteria were not met")
		os.Exit(1)
	}
	fmt.Println("RESULT: PASS - all acceptance criteria met")
}

func runFilters(s sample, medianParams, bilateralParams imaging.DenoiseParams) []filterResult {
	inputPSNR := nighteval.PSNR(s.clean, s.noisy)

	medianOut, medianMS := timed(func() *image.NRGBA {
		return imaging.MedianFilter(s.noisy, &medianParams)
	})
	bilateralOut, bilateralMS := timed(func() *image.NRGBA {
		return imaging.BilateralFilter(s.noisy, &bilateralParams)
	})

	return []filterResult{
		{
			name:          "median",
			output:        medianOut,
			inputPSNR:     inputPSNR,
			outputPSNR:    nighteval.PSNR(s.clean, medianOut),
			edgeRetention: nighteval.EdgeGradientRetention(s.clean, medianOut),
			blockRatio:    nighteval.BlockArtifactRetention(s.clean, medianOut),
			latencyMS:     medianMS,
		},
		{
			name:          "bilateral",
			output:        bilateralOut,
			inputPSNR:     inputPSNR,
			outputPSNR:    nighteval.PSNR(s.clean, bilateralOut),
			edgeRetention: nighteval.EdgeGradientRetention(s.clean, bilateralOut),
			blockRatio:    nighteval.BlockArtifactRetention(s.clean, bilateralOut),
			latencyMS:     bilateralMS,
		},
	}
}

func timed(fn func() *image.NRGBA) (*image.NRGBA, float64) {
	// One warm-up call so parallel goroutine setup does not skew measurements.
	_ = fn()
	start := time.Now()
	out := fn()
	elapsed := time.Since(start).Seconds() * 1000
	return out, elapsed
}

func benchmarkFilter(name string, img *image.NRGBA,
	fn func(image.Image) *image.NRGBA) float64 {
	const iterations = 3
	_ = fn(img) // warm-up
	var totalMS float64
	for i := 0; i < iterations; i++ {
		start := time.Now()
		_ = fn(img)
		totalMS += time.Since(start).Seconds() * 1000
	}
	avg := totalMS / iterations
	fmt.Printf("benchmark %s on %dx%d (%.1fMP): %.1f ms avg over %d runs\n",
		name, img.Bounds().Dx(), img.Bounds().Dy(),
		float64(img.Bounds().Dx()*img.Bounds().Dy())/1e6, avg, iterations)
	return avg
}

// scaleToPixelCount resizes the scene so that width*height is approximately
// targetPixels while preserving the 16:10 aspect ratio.
func scaleToPixelCount(src *image.NRGBA, targetPixels int) *image.NRGBA {
	srcW := src.Bounds().Dx()
	srcH := src.Bounds().Dy()
	ratio := float64(srcH) / float64(srcW)
	w := int(0.5 + math.Sqrt(float64(targetPixels)/ratio))
	h := int(0.5 + float64(w)*ratio)
	return imaging.Resize(src, w, h, imaging.Linear)
}

// buildComparison lays out [clean | noisy | median | bilateral] with a second
// row showing the center region at 2x zoom, plus label bars.
func buildComparison(s sample, results []filterResult) *image.NRGBA {
	panels := []*image.NRGBA{s.clean, s.noisy, results[0].output, results[1].output}
	labels := []string{"clean reference", "noisy input", "median (night default)", "bilateral (night default)"}

	labelH := 26
	panelW, panelH := sampleW/2, sampleH/2
	zoom := 2
	cropW, cropH := panelW/zoom, panelH/zoom

	canvasW := panelW * len(panels)
	canvasH := (labelH + panelH) * 2
	canvas := image.NewNRGBA(image.Rect(0, 0, canvasW, canvasH))
	fillCanvas(canvas, color.NRGBA{245, 245, 245, 255})

	for i, panel := range panels {
		x0 := i * panelW

		// Top row: full frame, downscaled.
		top := imaging.Resize(panel, panelW, panelH, imaging.Linear)
		draw.Draw(canvas, image.Rect(x0, labelH, x0+panelW, labelH+panelH),
			top, image.Point{}, draw.Src)
		drawLabel(canvas, x0+8, 18, labels[i])

		// Bottom row: center crop, 2x nearest-neighbor zoom (keeps pixels honest).
		cx, cy := panel.Bounds().Dx()/2, panel.Bounds().Dy()/2
		crop := imaging.Crop(panel, image.Rect(
			cx-cropW/2, cy-cropH/2, cx+cropW/2, cy+cropH/2))
		zoomed := imaging.Resize(crop, panelW, panelH, imaging.NearestNeighbor)
		draw.Draw(canvas, image.Rect(x0, 2*labelH+panelH, x0+panelW,
			2*labelH+2*panelH), zoomed, image.Point{}, draw.Src)
		drawLabel(canvas, x0+8, labelH+panelH+18, labels[i]+" (center 2x)")
	}
	return canvas
}

func fillCanvas(img *image.NRGBA, c color.NRGBA) {
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i] = c.R
		img.Pix[i+1] = c.G
		img.Pix[i+2] = c.B
		img.Pix[i+3] = c.A
	}
}

func drawLabel(img *image.NRGBA, x, y int, text string) {
	face := basicfont.Face7x13
	drawer := font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.NRGBA{20, 20, 20, 255}),
		Face: face,
		Dot:  fixed.P(x, y),
	}
	drawer.DrawString(text)
}

func savePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func boolMark(pass bool) string {
	if pass {
		return "PASS"
	}
	return "FAIL"
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
