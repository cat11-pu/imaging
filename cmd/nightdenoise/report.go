package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/disintegration/imaging"
	"github.com/disintegration/imaging/internal/nighteval"
)

func writeReport(outDir string, rows [][]string, median40MS, bilateral40MS float64,
	allPass bool, medianParams, bilateralParams imaging.DenoiseParams) error {
	var b strings.Builder

	b.WriteString("# 夜间监控降噪评估报告\n\n")
	b.WriteString("## 方法与默认参数（面向夜间监控调优）\n\n")
	b.WriteString(fmt.Sprintf("- 中值滤波 `MedianFilter`：radius=%d（%dx%d 窗口），对椒盐/脉冲噪声有效，输出值全部取自输入像素，不会产生块状平均伪影。\n",
		medianParams.Radius, 2*medianParams.Radius+1, 2*medianParams.Radius+1))
	b.WriteString(fmt.Sprintf("- 双边滤波 `BilateralFilter`：radius=%d, sigmaSpatial=%.1f, sigmaRange=%.1f，空间高斯核 + 亮度相似性高斯核（RGB 强相关的监控画面用亮度引导共享权重），平滑平坦区域的同时保留边缘，定点化 + 并行实现。\n",
		bilateralParams.Radius, bilateralParams.SigmaSpatial, bilateralParams.SigmaRange))
	b.WriteString(fmt.Sprintf("- 噪声模型：读取噪声高斯 sigma=%.1f + 信号相关散粒噪声 scale=%.2f + 椒盐脉冲 p=%.4f（模拟低照度传感器）。\n",
		nighteval.DefaultNightNoise.ReadSigma, nighteval.DefaultNightNoise.ShotScale, nighteval.DefaultNightNoise.ImpulseProb))
	b.WriteString("- 所有参数均通过 `DenoiseParams` 可调；传 `nil` 使用下列夜间默认值。\n\n")

	b.WriteString("## 指标定义与验收阈值\n\n")
	b.WriteString(fmt.Sprintf("- PSNR 提升 = 输出 PSNR − 加噪输入 PSNR，要求 ≥ %.0f dB（对干净参考图计算，RGB 三通道）。\n", minPSNRGainDB))
	b.WriteString(fmt.Sprintf("- 边缘梯度保持 = 在干净参考图前 10%% 强边缘位置上，滤波输出与干净参考的 Sobel 梯度强度均值之比，要求 ≥ %.2f；1.0 为完全保持，大于 1.0 表示输出比抗锯齿参考更锐利，低于该阈值才算糊边。\n", minEdgeRetention))
	b.WriteString(fmt.Sprintf("- 块效应比 = 滤波图 8 像素边界跳变能量/内部跳变能量，再除以干净参考的同一比值（消除场景固有周期结构），要求 ≤ %.2f，接近 1.0 表示未引入块结构。\n", maxBlockRatio))
	b.WriteString("- 单张耗时取预热后的单次测量；40MP 基准为 3 次平均，要求 ≤ 500 ms。\n\n")

	b.WriteString("## 三组样本结果\n\n")
	b.WriteString("对比图布局：[干净原图 | 加噪输入 | 中值滤波 | 双边滤波]，下排为中心区域 2x 放大（最近邻，保留真实像素）。\n\n")
	b.WriteString("| 样本组 | 滤波器 | 输入PSNR(dB) | 输出PSNR(dB) | 提升(dB) | 边缘梯度保持 | 块效应比 | 单张耗时(ms) | 结论 |\n")
	b.WriteString("|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rows {
		b.WriteString("| " + strings.Join(r, " | ") + " |\n")
	}

	b.WriteString("\n## 4000 万像素性能基准（16:10 场景，3 次平均）\n\n")
	b.WriteString("| 滤波器 | 单张耗时(ms) | 预算(ms) | 结论 |\n")
	b.WriteString("|---|---|---|---|\n")
	b.WriteString(fmt.Sprintf("| median | %.1f | %.0f | %s |\n",
		median40MS, max40MPLatencyMS, passMark(median40MS <= max40MPLatencyMS)))
	b.WriteString(fmt.Sprintf("| bilateral | %.1f | %.0f | %s |\n",
		bilateral40MS, max40MPLatencyMS, passMark(bilateral40MS <= max40MPLatencyMS)))

	b.WriteString("\n## 总体结论\n\n")
	if allPass && median40MS <= max40MPLatencyMS && bilateral40MS <= max40MPLatencyMS {
		b.WriteString("两种滤波器在三组样本上均满足 PSNR 提升 ≥ 6 dB、边缘梯度保持、无块效应要求，且 40MP 耗时均在 500 ms 预算内。\n")
	} else {
		b.WriteString("存在未达标项，请查看上表中的 FAIL 标记。\n")
	}

	path := filepath.Join(outDir, "report.md")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func passMark(pass bool) string {
	if pass {
		return "PASS"
	}
	return "FAIL"
}
