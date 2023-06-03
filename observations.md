Using ONLY the findBest profiling (aka ignoring any and all calls to clock 0)
we get the following profiling information:

- Avg. time used initializing shared memory: 6482
- Avg. time used on __syncthreads():         439438
- Avg. time used setting up the while loop:  16882188
- Avg. time used executing the while loop:   7222656
- Avg. time used calculating bundleScore:    5077586
- Avg. time used calculating seriesScore:    53395
- Avg. time used printing vals:              158698

If we record the profiling in the while loop as well, this is what we get:
- Avg. time used initializing shared memory: 6365
- Avg. time used on __syncthreads():         439489
- Avg. time used setting up the while loop:  18009274
- Avg. time used executing the while loop:   35253994
- Avg. time used calculating bundleScore:    5877519
- Avg. time used calculating seriesScore:    56483
- Avg. time used printing vals:              213516
- Loop times:
- Avg. time used checking loop condition:    4394792
- Avg. time used picking a set:              1907506
- Avg. time used validating set size:        11129702
- Avg. time used validating set bundles:     95117
- Avg. time used validating set is non-dupe: 418262
- Avg. time used adding set to DL:           23308

It makes sense that the while loop execution time is MUCH larger in the second case.
There is a lot of atomicAdd being called due to checkpoints.

However, an interesting observation is that, if the while loop profiling is ignored
(aka look at the first case) then setting up the while loop is the biggest time hog.