package nighteval

import (
	"image"
	"image/color"
	"math"
)

// fillRect paints a solid rectangle (inclusive border) on an NRGBA image.
func fillRect(img *image.NRGBA, x0, y0, x1, y1 int, c color.NRGBA) {
	b := img.Bounds()
	if x0 < b.Min.X {
		x0 = b.Min.X
	}
	if y0 < b.Min.Y {
		y0 = b.Min.Y
	}
	if x1 > b.Max.X-1 {
		x1 = b.Max.X - 1
	}
	if y1 > b.Max.Y-1 {
		y1 = b.Max.Y - 1
	}
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
}

// fillCircle paints a filled circle.
func fillCircle(img *image.NRGBA, cx, cy, radius int, c color.NRGBA) {
	for y := cy - radius; y <= cy+radius; y++ {
		for x := cx - radius; x <= cx+radius; x++ {
			dx := x - cx
			dy := y - cy
			if dx*dx+dy*dy <= radius*radius {
				img.SetNRGBA(x, y, c)
			}
		}
	}
}

// strokeLine paints a line with the given thickness using Bresenham-style steps.
func strokeLine(img *image.NRGBA, x0, y0, x1, y1, thickness int, c color.NRGBA) {
	dx := x1 - x0
	dy := y1 - y0
	steps := int(math.Max(math.Abs(float64(dx)), math.Abs(float64(dy))))
	if steps == 0 {
		fillCircle(img, x0, y0, thickness/2, c)
		return
	}
	for i := 0; i <= steps; i++ {
		t := float64(i) / float64(steps)
		x := int(math.Round(float64(x0) + float64(dx)*t))
		y := int(math.Round(float64(y0) + float64(dy)*t))
		fillCircle(img, x, y, thickness/2, c)
	}
}

// NightStatic renders a static night surveillance scene: a dark street with a
// building, lit windows, a fence and vegetation. Large flat dark areas dominate
// the frame so that night noise is visually obvious.
func NightStatic(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))

	for y := 0; y < h; y++ {
		t := float64(y) / float64(h)
		// Vertical night-sky gradient (very dark blue).
		c := color.NRGBA{
			R: uint8(6 + 10*t),
			G: uint8(10 + 12*t),
			B: uint8(22 + 18*t),
			A: 255,
		}
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, c)
		}
	}

	// Ground / road.
	fillRect(img, 0, h*2/3, w, h, color.NRGBA{26, 28, 30, 255})
	fillRect(img, 0, h*3/4, w, h, color.NRGBA{20, 21, 23, 255})

	// Building on the left.
	fillRect(img, w/16, h/6, w*7/16, h*2/3, color.NRGBA{44, 46, 54, 255})
	windowColor := color.NRGBA{214, 178, 110, 255}
	dimWindow := color.NRGBA{96, 84, 60, 255}
	for wy := h / 5; wy < h*3/5; wy += h / 9 {
		for wx := w / 12; wx < w*7/16-w/24; wx += w / 10 {
			c := windowColor
			if (wx+wy)%3 == 0 {
				c = dimWindow
			}
			fillRect(img, wx, wy, wx+w/28, wy+h/22, c)
		}
	}

	// Fence with vertical bars (thin high-contrast edges).
	for x := w / 2; x < w*15/16; x += w / 26 {
		fillRect(img, x, h/2, x+2, h*3/4, color.NRGBA{70, 72, 78, 255})
	}
	strokeLine(img, w/2, h/2, w*15/16, h/2, 3, color.NRGBA{70, 72, 78, 255})

	// Tree silhouettes.
	fillRect(img, w*5/6-3, h/3, w*5/6+3, h*3/4, color.NRGBA{16, 18, 16, 255})
	fillCircle(img, w*5/6, h/4, h/10, color.NRGBA{18, 24, 18, 255})
	fillCircle(img, w*5/6-h/14, h/4+h/16, h/14, color.NRGBA{16, 22, 16, 255})

	// Road lane marking (a static thin bright line).
	for x := w / 12; x < w; x += w / 8 {
		fillRect(img, x, h*7/8, x+w/16, h*7/8+3, color.NRGBA{150, 148, 130, 255})
	}

	return img
}

// NightDynamic renders a night scene containing a moving vehicle. The vehicle
// and its headlight beams are directionally blurred to simulate motion; the
// background stays identical in spirit to the static scene.
func NightDynamic(w, h int) *image.NRGBA {
	img := NightStatic(w, h)

	// Headlight beams on the road (wide soft wedges).
	beam := color.NRGBA{120, 116, 90, 255}
	strokeLine(img, w/3, h*2/3, 0, h*9/10, 10, beam)
	strokeLine(img, w/3, h*2/3+12, w/12, h*9/10, 8, color.NRGBA{90, 88, 70, 255})

	// Vehicle body, horizontally smeared over a few pixels to suggest motion.
	carTop := color.NRGBA{52, 58, 70, 255}
	carBody := color.NRGBA{38, 42, 52, 255}
	cx0, cy0, cx1, cy1 := w/6, h/2, w*4/9, h*2/3
	for step := 0; step < 6; step++ {
		shift := step * 2
		fillRect(img, cx0+shift, cy0+8, cx1+shift, cy1, carBody)
		fillRect(img, cx0+w/18+shift, cy0, cx1-w/14+shift, cy0+8, carTop)
	}

	// Windshield.
	fillRect(img, cx0+w/14, cy0+2, cx1-w/12, cy0+7, color.NRGBA{96, 116, 130, 255})

	// Headlights and tail lights.
	fillCircle(img, cx1-2, cy1-6, 5, color.NRGBA{235, 226, 180, 255})
	fillCircle(img, cx0+2, cy1-6, 4, color.NRGBA{180, 40, 30, 255})

	// Wheels.
	wheel := color.NRGBA{12, 12, 12, 255}
	fillCircle(img, cx0+w/14, cy1, 9, wheel)
	fillCircle(img, cx1-w/14, cy1, 9, wheel)

	return img
}

// LowlightForensics renders an extremely dark scene packed with fine,
// high-contrast detail (resolution wedges, grids, text-like bars and step
// edges), matching a low-illumination forensic capture where detail and edges
// must survive denoising.
func LowlightForensics(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))

	// Nearly black, slightly tinted background.
	fillRect(img, 0, 0, w, h, color.NRGBA{14, 15, 18, 255})

	light := color.NRGBA{118, 122, 128, 255}
	faint := color.NRGBA{52, 55, 60, 255}

	// Checkerboard with 9px cells (coarser than the 5x5 median window so the
	// structure survives) and non-8-aligned placement.
	cell := 9
	cx0, cy0 := w/14, h/7
	cols, rows := 12, 9
	for gy := 0; gy < rows; gy++ {
		for gx := 0; gx < cols; gx++ {
			if (gx+gy)%2 == 0 {
				fillRect(img, cx0+gx*cell, cy0+gy*cell,
					cx0+(gx+1)*cell-1, cy0+(gy+1)*cell-1, faint)
			}
		}
	}

	// Step-edge staircase of increasing brightness (for gradient checks),
	// each step wide enough to remain a clean edge after median filtering.
	stepW := w / 16
	for i := 0; i < 7; i++ {
		v := uint8(26 + i*14)
		fillRect(img, w/2+i*stepW, h*3/4, w/2+(i+1)*stepW, h*7/8,
			color.NRGBA{v, v, v + 2, 255})
	}

	// Bold alternating bars resembling a forensic label (6px strokes).
	for i := 0; i < 12; i++ {
		bw := 6 + (i%2)*4
		x := w/2 + i*14
		fillRect(img, x, h/10, x+bw, h/10+h/14, light)
	}

	// Circular evidence target with thick rings.
	fillCircle(img, w*3/4, h/3, h/10, faint)
	fillCircle(img, w*3/4, h/3, h/15, color.NRGBA{14, 15, 18, 255})
	fillCircle(img, w*3/4, h/3, h/24, light)

	// Reflective evidence markers (small but multi-pixel so they survive).
	fillCircle(img, w/6, h*5/6, 3, light)
	fillCircle(img, w*2/7, h*11/12, 3, light)
	fillCircle(img, w/3, h*5/7, 2, light)

	return img
}
