#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::collections::BTreeSet;

use bitset_fixed::BitSet;
use itertools::{max, Itertools};
use ndarray::IntoDimension;
use proconio::{input, marker::Usize1};
use rand::prelude::*;
use superslice::*;

fn main() {
    let start = std::time::Instant::now();

    solve();

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

fn solve() {
    let input = read_input();
    let mut out = vec![vec![(0, 0, 0, 0); input.N]; input.D];
    for d in 0..input.D {
        let mut now_i = 0;
        let mut now_j = 0;
        let mut width = input.W;
        let mut height = input.W;
        let mut used = vec![false; input.N];

        for i in 0..input.N {
            let mut cands = vec![];
            for n in 0..input.N {
                if used[n] {
                    continue;
                }
                let chunk = width;
                let len = ceil(input.A[d][n], chunk);
                cands.push((chunk * len - input.A[d][n], n, chunk, len));
                let chunk = height;
                let len = ceil(input.A[d][n], chunk);
                cands.push((chunk * len - input.A[d][n], n, chunk, len));
            }
            cands.sort();
            let (_, n, chunk, len) = cands[0];
            used[n] = true;
            if chunk == height {
                if width < len {
                    break;
                }
                out[d][n] = (now_i, now_j, now_i + chunk, now_j + len);
                now_j += len;
                width -= len;
            } else {
                if height < len {
                    break;
                }
                out[d][n] = (now_i, now_j, now_i + len, now_j + chunk);
                now_i += len;
                height -= len;
            }
            if i == input.N - 1 {
                out[d][n].2 = input.W;
                out[d][n].3 = input.W;
            }
        }
    }

    let mut density = vec![];
    for d in 0..input.D {
        let mut sum = 0;
        for n in 0..input.N {
            sum += input.A[d][n];
        }
        density.push(sum as f64 / 1e6);
    }
    let ave = density.iter().sum::<f64>() / density.len() as f64;
    if ave > 0.975 {
        output(&input, &out);
        return;
    }

    let time_limit = 2.9;
    let time_keeper = TimeKeeper::new(time_limit);
    let mut rng = rand_pcg::Pcg64Mcg::new(0);

    let mut order = (0..input.N).collect_vec();
    let mut out = vec![vec![(0, 0, 0, 0); input.N]; input.D];

    let mut best_out = out.clone();
    let mut best_score = 0;

    while !time_keeper.isTimeOver() {
        let mut now_i = 0;
        let mut now_j = 0;
        let mut width = input.W;
        let mut height = input.W;

        let mut track = vec![];

        for i in 0..input.N {
            let n = order[i];
            if height == 0 || width == 0 {
                break;
            }
            let mut mx_height = 0;
            let mut mx_width = 0;
            for d in 0..input.D {
                mx_height.chmax(ceil(input.A[d][n], height));
                mx_width.chmax(ceil(input.A[d][n], width));
            }
            let (chunk, mx) = {
                if mx_height * height < mx_width * width {
                    (height, mx_height)
                } else {
                    (width, mx_width)
                }
            };

            if chunk == height {
                if width < mx {
                    break;
                }
                now_j += mx;
                width -= mx;
            } else {
                if height < mx {
                    break;
                }
                now_i += mx;
                height -= mx;
            }
            track.push((now_i, now_j, chunk, mx));
        }

        let mut prefix_sum = vec![vec![0; input.N + 1]; input.D];
        for d in 0..input.D {
            for i in 0..input.N {
                let n = order[i];
                prefix_sum[d][i + 1] = prefix_sum[d][i] + input.A[d][n];
            }
        }
        let mut suffix_max = vec![vec![0; input.N]; input.D];
        for d in (0..input.D).rev() {
            for n in 0..input.N {
                suffix_max[d][n].chmax(input.A[d][n]);
                if d != input.D - 1 {
                    let after = suffix_max[d + 1][n];
                    suffix_max[d][n].chmax(after);
                }
            }
        }

        let mut track_cand = vec![vec![]; input.D];
        let mut score = 1e18 as usize;

        for d in 0..input.D {
            let mut sum = 0;
            let mut now_i = 0;
            let mut now_j = 0;
            let mut latch = false;
            for i in 0..input.N {
                let n = order[i];
                if d == 0 {
                    if i < track.len() {
                        let (now_ii, now_jj, chunk, len) = track[i];
                        now_i = now_ii;
                        now_j = now_jj;
                        sum += chunk * len;
                        track_cand[d].push((sum, now_i, now_j, chunk, len));
                    } else {
                        let height = input.W - now_i;
                        let width = input.W - now_j;
                        if height == 0 || width == 0 {
                            break;
                        }
                        let len_height = ceil(input.A[d][n], height);
                        let len_width = ceil(input.A[d][n], width);
                        let (chunk, len) = {
                            if len_height * height < len_width * width {
                                (height, len_height)
                            } else {
                                (width, len_width)
                            }
                        };
                        sum += chunk * len;
                        if chunk == height {
                            now_j += len;
                        } else {
                            now_i += len;
                        }
                        track_cand[d].push((sum, now_i, now_j, chunk, len));
                    }
                } else {
                    let (s, now_ii, now_jj, chunk, len) = track_cand[d - 1][i];
                    if latch || suffix_max[d][n] > chunk * len {
                        latch = true;
                        let height = input.W - now_i;
                        let width = input.W - now_j;
                        if height == 0 || width == 0 {
                            break;
                        }
                        let len_height = ceil(suffix_max[d][n], height);
                        let len_width = ceil(suffix_max[d][n], width);
                        let (chunk2, len2) = {
                            if len_height * height < len_width * width {
                                (height, len_height)
                            } else {
                                (width, len_width)
                            }
                        };
                        sum += chunk2 * len2;
                        if chunk2 == height {
                            now_j += len2;
                        } else {
                            now_i += len2;
                        }
                        track_cand[d].push((sum, now_i, now_j, chunk2, len2));
                    } else {
                        score -= 1;
                        now_i = now_ii;
                        now_j = now_jj;
                        sum = s;
                        track_cand[d].push((sum, now_i, now_j, chunk, len));
                    }
                }
                if sum > input.W * input.W {
                    break;
                }
            }

            let L = track_cand[d].len();
            if track_cand[d][L - 1].0 <= input.W * input.W && now_i <= input.W && now_j <= input.W {
                continue;
            }

            'a: for i in (0..input.N).rev() {
                let mut now_i = 0;
                let mut now_j = 0;
                for ii in i..input.N {
                    let n = order[ii];
                    let (s, now_ii, now_jj, _, _) = if ii == 0 {
                        (0, 0, 0, 0, 0)
                    } else {
                        if ii > track_cand[d].len() {
                            continue 'a;
                        }
                        track_cand[d][ii - 1]
                    };
                    if now_ii > input.W || now_jj > input.W {
                        continue 'a;
                    }
                    let height = input.W - now_ii;
                    let width = input.W - now_jj;
                    if height == 0 || width == 0 {
                        continue 'a;
                    }
                    let len_height = ceil(input.A[d][n], height);
                    let len_width = ceil(input.A[d][n], width);
                    let (chunk, len) = {
                        if len_height * height < len_width * width {
                            (height, len_height)
                        } else {
                            (width, len_width)
                        }
                    };
                    if s + chunk * len > input.W * input.W {
                        continue 'a;
                    }
                    if chunk == height {
                        if now_jj + len > input.W {
                            continue 'a;
                        }
                        if ii >= track_cand[d].len() {
                            track_cand[d].push((0, 0, 0, 0, 0));
                        }
                        track_cand[d][ii] = (s + chunk * len, now_ii, now_jj + len, chunk, len);
                        now_i = now_ii;
                        now_j = now_jj + len;
                    } else {
                        if now_ii + len > input.W {
                            continue 'a;
                        }
                        if ii >= track_cand[d].len() {
                            track_cand[d].push((0, 0, 0, 0, 0));
                        }
                        track_cand[d][ii] = (s + chunk * len, now_ii + len, now_jj, chunk, len);
                        now_i = now_ii + len;
                        now_j = now_jj;
                    }
                }
                if track_cand[d][input.N - 1].0 <= input.W * input.W
                    && now_i <= input.W
                    && now_j <= input.W
                {
                    break;
                }
            }
        }

        for d in 0..input.D {
            let mut now_i = 0;
            let mut now_j = 0;
            if track_cand[d].len() != input.N {
                break;
            }
            for i in 0..input.N {
                let n = order[i];
                let (_, next_i, next_j, chunk, _) = track_cand[d][i];
                if now_i == next_i {
                    out[d][n] = (now_i, now_j, now_i + chunk, next_j);
                } else {
                    out[d][n] = (now_i, now_j, next_i, now_j + chunk);
                }
                now_i = next_i;
                now_j = next_j;
            }
        }

        for d in 0..input.D {
            out[d][order[input.N - 1]].2 = input.W;
            out[d][order[input.N - 1]].3 = input.W;
        }

        if score > best_score {
            best_score = score;
            best_out = out.clone();
        }
        let swap_i = rng.gen_range(0..input.N - 1);
        order.swap(swap_i, swap_i + 1);
    }
    output(&input, &best_out);
}

fn output(input: &Input, out: &[Vec<(usize, usize, usize, usize)>]) {
    for d in 0..input.D {
        for n in 0..input.N {
            println!(
                "{} {} {} {}",
                out[d][n].0, out[d][n].1, out[d][n].2, out[d][n].3
            );
        }
    }

    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        let mut score = compute_score_details(&input, &out);
        if score == 0 {
            score = 1e9 as i64;
        }
        eprintln!("Score: {}", score);
    }
}

fn ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn compute_score_details(input: &Input, out: &[Vec<(usize, usize, usize, usize)>]) -> i64 {
    let mut score = 0;
    let mut change: Vec<(usize, usize, usize, usize, bool)> = vec![];
    let mut hs = BTreeSet::new();
    let mut vs = BTreeSet::new();
    for d in 0..out.len() {
        for p in 0..input.N {
            for q in 0..p {
                if out[d][p].2.min(out[d][q].2) > out[d][p].0.max(out[d][q].0)
                    && out[d][p].3.min(out[d][q].3) > out[d][p].1.max(out[d][q].1)
                {
                    return 0;
                }
            }
        }
        let mut hs2 = BTreeSet::new();
        let mut vs2 = BTreeSet::new();
        for k in 0..input.N {
            let (i0, j0, i1, j1) = out[d][k];
            let b = (i1 - i0) * (j1 - j0);
            if input.A[d][k] > b {
                score += 100 * (input.A[d][k] - b) as i64;
            }
            for j in j0..j1 {
                if i0 > 0 {
                    hs2.insert((i0, j));
                }
                if i1 < input.W {
                    hs2.insert((i1, j));
                }
            }
            for i in i0..i1 {
                if j0 > 0 {
                    vs2.insert((j0, i));
                }
                if j1 < input.W {
                    vs2.insert((j1, i));
                }
            }
        }
        if d > 0 {
            for &(i, j) in &hs {
                if !hs2.contains(&(i, j)) {
                    score += 1;
                    if d + 1 == out.len() {
                        if !change.is_empty()
                            && change[change.len() - 1]
                                == (i, change[change.len() - 1].1, i, j, false)
                        {
                            change.last_mut().unwrap().3 += 1;
                        } else {
                            change.push((i, j, i, j + 1, false));
                        }
                    }
                }
            }
            for &(j, i) in &vs {
                if !vs2.contains(&(j, i)) {
                    score += 1;
                    if d + 1 == out.len() {
                        if !change.is_empty()
                            && change[change.len() - 1]
                                == (change[change.len() - 1].0, j, i, j, false)
                        {
                            change.last_mut().unwrap().2 += 1;
                        } else {
                            change.push((i, j, i + 1, j, false));
                        }
                    }
                }
            }
            for &(i, j) in &hs2 {
                if !hs.contains(&(i, j)) {
                    score += 1;
                    if d + 1 == out.len() {
                        if !change.is_empty()
                            && change[change.len() - 1]
                                == (i, change[change.len() - 1].1, i, j, true)
                        {
                            change.last_mut().unwrap().3 += 1;
                        } else {
                            change.push((i, j, i, j + 1, true));
                        }
                    }
                }
            }
            for &(j, i) in &vs2 {
                if !vs.contains(&(j, i)) {
                    score += 1;
                    if d + 1 == out.len() {
                        if !change.is_empty()
                            && change[change.len() - 1]
                                == (change[change.len() - 1].0, j, i, j, true)
                        {
                            change.last_mut().unwrap().2 += 1;
                        } else {
                            change.push((i, j, i + 1, j, true));
                        }
                    }
                }
            }
        }
        hs = hs2;
        vs = vs2;
    }
    score + 1
}

struct Input {
    W: usize,
    D: usize,
    N: usize,
    A: Vec<Vec<usize>>,
}

fn read_input() -> Input {
    input! {
        W: usize,
        D: usize,
        N: usize,
        A: [[usize; N]; D]
    }

    Input { W, D, N, A }
}

pub trait ChangeMinMax {
    fn chmin(&mut self, x: Self) -> bool;
    fn chmax(&mut self, x: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn chmin(&mut self, x: Self) -> bool {
        *self > x && {
            *self = x;
            true
        }
    }
    fn chmax(&mut self, x: Self) -> bool {
        *self < x && {
            *self = x;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct TimeKeeper {
    start_time: std::time::Instant,
    time_threshold: f64,
}

impl TimeKeeper {
    fn new(time_threshold: f64) -> Self {
        TimeKeeper {
            start_time: std::time::Instant::now(),
            time_threshold,
        }
    }
    #[inline]
    fn isTimeOver(&self) -> bool {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55 >= self.time_threshold
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time >= self.time_threshold
        }
    }
    #[inline]
    fn get_time(&self) -> f64 {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}
