package imaging

import (
	"runtime"
	"sync"
)

// defaultChunkRows is the default number of rows (or columns) a single
// scheduled work unit covers. Coarse chunks keep the per-task scheduling
// overhead negligible even for small images.
const defaultChunkRows = 64

// rangeFunc processes every index in the half-open interval [start, stop).
type rangeFunc func(start, stop int)

// chunkJob is one contiguous [start, stop) range of indices processed by a
// single pool worker. Ranges never overlap, so every output pixel is written
// by exactly one goroutine and the result is identical to a single-threaded
// run.
type chunkJob struct {
	start int
	stop  int
	fn    rangeFunc
	wg    *sync.WaitGroup
}

// workerPool is a fixed-size pool of long-lived goroutines reused across all
// image processing calls (and therefore across whole batch tasks). Workers
// block on the shared jobs channel between calls instead of being recreated.
type workerPool struct {
	mu      sync.Mutex
	workers int
	jobs    chan chunkJob
	wg      sync.WaitGroup
}

// ensure resizes the pool to n workers. Workers are created once and stay
// blocked on the shared jobs channel between calls; shrinking retires surplus
// workers by sending them nil (quit) jobs.
func (p *workerPool) ensure(n int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if n < 1 {
		n = 1
	}
	if n == p.workers && p.jobs != nil {
		return
	}
	if p.jobs == nil {
		p.jobs = make(chan chunkJob)
	}

	for p.workers < n {
		p.workers++
		p.wg.Add(1)
		go p.run()
	}
	for p.workers > n {
		p.jobs <- chunkJob{}
		p.workers--
	}
}

func (p *workerPool) run() {
	defer p.wg.Done()
	// The channel is shared by every parallel call. Each worker handles one
	// contiguous chunk, reports it on the call's WaitGroup and asks for more.
	for job := range p.jobs {
		if job.fn == nil {
			return
		}
		job.fn(job.start, job.stop)
		job.wg.Done()
	}
}

// poolConfig holds the user-tunable parallelism settings. Non-positive values
// mean "use the default".
type poolConfig struct {
	mu      sync.RWMutex
	workers int
	chunk   int
}

var config poolConfig

var pool workerPool

// defaultWorkers returns half of the available CPU cores, at least one.
func defaultWorkers() int {
	n := runtime.NumCPU() / 2
	if n < 1 {
		n = 1
	}
	return n
}

// workerCount resolves the effective worker count: the configured value when
// positive, otherwise half of the CPU cores.
func workerCount() int {
	config.mu.RLock()
	n := config.workers
	config.mu.RUnlock()
	if n > 0 {
		return n
	}
	return defaultWorkers()
}

func chunkSize() int {
	config.mu.RLock()
	n := config.chunk
	config.mu.RUnlock()
	if n > 0 {
		return n
	}
	return defaultChunkRows
}

// SetWorkers sets the number of worker goroutines used for parallel image
// processing. A value <= 0 restores the default, which is half of the CPU
// cores (at least one). The underlying worker pool is resized lazily and its
// workers are reused across subsequent calls and whole batch tasks.
func SetWorkers(value int) {
	config.mu.Lock()
	config.workers = value
	config.mu.Unlock()
}

// SetMaxProcs limits the number of worker goroutines to the given value.
// A value <= 0 restores the default (half of the CPU cores).
//
// Deprecated: use SetWorkers, which provides the same configuration with a
// clearer name and documented default behavior.
func SetMaxProcs(value int) {
	SetWorkers(value)
}

// SetChunkSize sets the number of rows (or columns) covered by one scheduled
// work unit. Larger chunks reduce scheduling overhead, which matters for
// small images; smaller chunks improve load balancing for uneven workloads.
// A value <= 0 restores the default of 64.
func SetChunkSize(rows int) {
	config.mu.Lock()
	config.chunk = rows
	config.mu.Unlock()
}

// parallel splits the [start, stop) index range into fixed-size contiguous
// chunks and dispatches them onto the reusable worker pool. Each index is
// processed exactly once and chunks never overlap, so the output is
// byte-identical to processing every index sequentially in one goroutine.
func parallel(start, stop int, fn rangeFunc) {
	count := stop - start
	if count < 1 {
		return
	}

	procs := workerCount()
	if procs > count {
		procs = count
	}

	chunk := chunkSize()
	if chunk < 1 {
		chunk = 1
	}

	// One worker, or a single chunk: scheduling cannot help, so run the work
	// inline and pay no channel round-trip or pool synchronization cost. This
	// keeps tiny images free of parallel overhead.
	if procs < 2 || count <= chunk {
		fn(start, stop)
		return
	}

	pool.ensure(procs)

	chunkCount := (count + chunk - 1) / chunk
	var wg sync.WaitGroup
	wg.Add(chunkCount)
	jobs := make([]chunkJob, chunkCount)
	for i := range jobs {
		s := start + i*chunk
		e := s + chunk
		if e > stop {
			e = stop
		}
		jobs[i] = chunkJob{start: s, stop: e, fn: fn, wg: &wg}
	}

	for i := range jobs {
		pool.jobs <- jobs[i]
	}
	wg.Wait()
}
