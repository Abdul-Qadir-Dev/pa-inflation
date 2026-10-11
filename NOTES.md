2026-10-11

- The NASA S3 bucket for telemanom returns 403 Forbidden, from both requests and curl and on both URL styles. The labels CSV on GitHub still downloads fine, but there's no data to go with it. Going ahead with NAB only. The loader for SMAP/MSL stays in windows.py in case the bucket comes back.
- All 34 NAB series in the three real folders downloaded on the first try. I'm using the keys in combined_windows.json as the file list, which saved me from calling the GitHub API.
- Four series have no labelled window in their second half (ec2_cpu_utilization_825cc2, ec2_cpu_utilization_c6585a, ec2_network_in_257a54, iio NetworkIn), so F1 isn't defined on the test part. Dropping them leaves 30.
- A lot of series have labelled anomalies in the first half too, which means the "train" data isn't clean. That's the NAB split problem and I can't do much about it without cherry-picking series.
- Test scoring borrows the last 63 training points as context so every test point gets a window ending on it. That keeps scores and labels the same length with no padding.
- First plumbing run on three AWS series. Random scores already get PA-F1 around 0.95. Their plain F1 is about 0.19, which is basically the flag-everything score 2p/(1+p) for a 10% anomaly rate. So the best-threshold sweep falls back to "flag it all" when nothing works.
- LSTM AE first pass on machine_temperature. Training windows with stride 1 took 39s per model against 15s with stride 8, and stride 1 actually scored a bit worse (AUROC 0.75 vs 0.84). With 30 series and 3 seeds that's the difference between an hour and twenty minutes, so stride 8 stays.
- Same series, a funny one already: the LSTM AE beats the z-score on plain F1 (0.40 vs 0.20) but loses on PA-F1 (0.78 vs 0.96). The z-score fires on a handful of points in each window, and point adjustment turns that into full credit.
