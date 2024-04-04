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
    let time_limit = 2.9;
    let time_keeper = TimeKeeper::new(time_limit);
    let mut rng = rand_pcg::Pcg64Mcg::new(0);
    let input = read_input();

    // let T0 = input.W as f64 * 2.0 * input.N as f64 / 5.0;
    // let T1 = input.W as f64 / 2.0;

    let mut order = (0..input.N).collect_vec();
    let mut out = vec![vec![(0, 0, 0, 0); input.N]; input.D];
    let mut chunks = vec![vec![0; input.N]; input.D];
    let cost = std::usize::MAX;

    let mut iter = 0;
    // let mut swap_i = 0;
    // let mut swap_j = 0;

    let mut best_out = out.clone();
    let mut best_cost = cost;

    while !time_keeper.isTimeOver() {
        iter += 1;

        let mut now_i = 0;
        let mut now_j = 0;
        let mut width = input.W;
        let mut height = input.W;

        let mut track = vec![(now_i, now_j)];

        for i in 0..input.N {
            let n = order[i];
            let chunk = if height < width { height } else { width };
            if chunk == 0 {
                break;
            }
            let mut mx = 0;
            // let mut mx_height = 0;
            // let mut mx_width = 0;
            for d in 0..input.D {
                // mx_height.chmax(ceil(input.A[d][n], height));
                // mx_width.chmax(ceil(input.A[d][n], width));
                mx.chmax(ceil(input.A[d][n], chunk));
            }
            // let (chunk, mx) = {
            //     if mx_height * height < mx_width * width {
            //         (height, mx_height)
            //     } else {
            //         (width, mx_width)
            //     }
            // };

            if chunk == height {
                if width < mx {
                    break;
                }
                for d in 0..input.D {
                    out[d][n] = (now_i, now_j, now_i + chunk, now_j + mx);
                }
                now_j += mx;
                width -= mx;
            } else {
                if height < mx {
                    break;
                }
                for d in 0..input.D {
                    out[d][n] = (now_i, now_j, now_i + mx, now_j + chunk);
                }
                now_i += mx;
                height -= mx;
            }
            track.push((now_i, now_j));
            for d in 0..input.D {
                chunks[d][i] = chunk;
            }
        }
        // for (n, (i, j)) in track.iter().enumerate().rev() {
        //     eprintln!("n: {} i: {} j: {}", n, i, j);
        // }

        let mut prefix_sum = vec![vec![0; input.N + 1]; input.D];
        for d in 0..input.D {
            for i in 0..input.N {
                let n = order[i];
                prefix_sum[d][i + 1] = prefix_sum[d][i] + input.A[d][n];
            }
        }

        let mut fixed_right = vec![0; input.D];

        for d in 0..input.D {
            for (k, (i, j)) in track.iter().enumerate().rev() {
                let margin = (input.W - i) * (input.W - j);
                let necessary = prefix_sum[d][input.N] - prefix_sum[d][k];
                if necessary > margin {
                    continue;
                }

                let mut now_i = *i;
                let mut now_j = *j;
                let mut height = input.W - i;
                let mut width = input.W - j;
                let mut ok = true;

                for ii in k..input.N {
                    let nn = order[ii];
                    let chunk = if height < width { height } else { width };
                    if chunk == 0 {
                        ok = false;
                        break;
                    }
                    let len = ceil(input.A[d][nn], chunk);

                    // let len_height = ceil(input.A[d][nn], height);
                    // let len_width = ceil(input.A[d][nn], width);
                    // let (chunk, len) = {
                    //     if len_height * height < len_width * width {
                    //         (height, len_height)
                    //     } else {
                    //         (width, len_width)
                    //     }
                    // };

                    if chunk == height {
                        if width < len {
                            ok = false;
                            break;
                        }
                        out[d][nn] = (now_i, now_j, now_i + chunk, now_j + len);
                        now_j += len;
                        width -= len;
                    } else {
                        if height < len {
                            ok = false;
                            break;
                        }
                        out[d][nn] = (now_i, now_j, now_i + len, now_j + chunk);
                        now_i += len;
                        height -= len;
                    }
                    chunks[d][ii] = chunk;
                }
                if ok {
                    fixed_right[d] = k;
                    break;
                }
            }
        }

        for d in 0..input.D {
            out[d][order[input.N - 1]].2 = input.W;
            out[d][order[input.N - 1]].3 = input.W;
        }

        let mut next_cost = 0;
        for d in 1..input.D {
            for i in fixed_right[d - 1].min(fixed_right[d])..input.N - 1 {
                next_cost += chunks[d - 1][i];
                next_cost += chunks[d][i];
            }
        }

        if next_cost < best_cost {
            best_cost = next_cost;
            best_out = out.clone();
        }

        // let diff = next_cost as isize - cost as isize;
        // let temp = T0 + (T1 - T0) * time_keeper.get_time() / time_limit;
        // if diff <= 0 || rng.gen_bool((-diff as f64 / temp).exp()) {
        //     cost = next_cost;
        // } else {
        //     order.swap(swap_i, swap_j);
        // }

        let swap_i = rng.gen_range(0..input.N - 1);
        // swap_j = swap_i + 1;
        // while swap_i == swap_j {
        //     swap_j = rng.gen_range(0..input.N);
        // }
        // order.swap(swap_i, swap_j);
        order.swap(swap_i, swap_i + 1);
    }

    for d in 0..input.D {
        for n in 0..input.N {
            println!(
                "{} {} {} {}",
                best_out[d][n].0, best_out[d][n].1, best_out[d][n].2, best_out[d][n].3
            );
        }
    }

    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        eprintln!("Iteration: {}", iter);
        let mut score = compute_score_details(&input, &best_out);
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
    let mut score1 = 0;
    let mut score2 = 0;
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
                score1 += 100 * (input.A[d][k] - b) as i64;
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
                    score2 += 1;
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
                    score2 += 1;
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
                    score2 += 1;
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
                    score2 += 1;
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
    assert!(score == score1 + score2);
    // eprintln!("{} {} {}", score, score1, score2);
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