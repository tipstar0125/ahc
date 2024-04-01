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
use rustc_hash::FxHashSet;
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
        eprintln!("state: 1");
        return;
    }

    let time_limit = 2.9;
    let time_keeper = TimeKeeper::new(time_limit);
    let mut rng = rand_pcg::Pcg64Mcg::new(0);

    let mut order1 = (0..input.N).rev().collect_vec();
    let mut out1 = vec![vec![(0, 0, 0, 0); input.N]; input.D];
    let mut chunks1 = vec![vec![0; input.N]; input.D];

    let mut best_out1 = out1.clone();
    let mut best_cost1 = std::usize::MAX;

    while time_keeper.get_time() <= 1.0 {
        let cost = f(&input, &order1, &mut chunks1, &mut out1, 0, input.D);

        if cost < best_cost1 {
            best_cost1 = cost;
            best_out1 = out1.clone();
        }

        let swap_i = rng.gen_range(0..input.N - 1);
        order1.swap(swap_i, swap_i + 1);
    }

    let mut order2 = (0..input.N).rev().collect_vec();
    let mut out2 = vec![vec![(0, 0, 0, 0); input.N]; input.D];
    let mut chunks2 = vec![vec![0; input.N]; input.D];

    let mut best_out2 = out2.clone();
    let mut bc2 = std::usize::MAX;

    while time_keeper.get_time() <= 1.5 {
        let cost = f(&input, &order2, &mut chunks2, &mut out2, 0, input.D / 2);

        if cost < bc2 {
            bc2 = cost;
            best_out2 = out2.clone();
        }

        let swap_i = rng.gen_range(0..input.N - 1);
        order2.swap(swap_i, swap_i + 1);
    }
    let mut bc2 = std::usize::MAX;
    while time_keeper.get_time() <= 2.0 {
        let cost = f(
            &input,
            &order2,
            &mut chunks2,
            &mut out2,
            input.D / 2,
            input.D,
        );

        if cost < bc2 {
            bc2 = cost;
            best_out2 = out2.clone();
        }

        let swap_i = rng.gen_range(0..input.N - 1);
        order2.swap(swap_i, swap_i + 1);
    }

    if compute_score_details(&input, &best_out1) < compute_score_details(&input, &best_out2) {
        output(&input, &best_out1);
        eprintln!("state: 2");
    } else {
        output(&input, &best_out2);
        eprintln!("state: 3");
    }
}

fn f(
    input: &Input,
    order: &[usize],
    chunks: &mut [Vec<usize>],
    out: &mut [Vec<(usize, usize, usize, usize)>],
    l: usize,
    r: usize,
) -> usize {
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
        let mut mx_height = 0;
        let mut mx_width = 0;
        for d in l..r {
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
            for d in l..r {
                out[d][n] = (now_i, now_j, now_i + chunk, now_j + mx);
            }
            now_j += mx;
            width -= mx;
        } else {
            if height < mx {
                break;
            }
            for d in l..r {
                out[d][n] = (now_i, now_j, now_i + mx, now_j + chunk);
            }
            now_i += mx;
            height -= mx;
        }
        track.push((now_i, now_j));
        for d in l..r {
            chunks[d][i] = chunk;
        }
    }

    let mut prefix_sum = vec![vec![0; input.N + 1]; input.D];
    for d in l..r {
        for i in 0..input.N {
            let n = order[i];
            prefix_sum[d][i + 1] = prefix_sum[d][i] + input.A[d][n];
        }
    }

    let mut fixed_right = vec![std::usize::MAX; input.D];

    for d in l..r {
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

                let len_height = ceil(input.A[d][nn], height);
                let len_width = ceil(input.A[d][nn], width);
                let (chunk, len) = {
                    if len_height * height < len_width * width {
                        (height, len_height)
                    } else {
                        (width, len_width)
                    }
                };

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

    for d in l..r {
        out[d][order[input.N - 1]].2 = input.W;
        out[d][order[input.N - 1]].3 = input.W;
    }

    let mut cost = 0;
    for d in l + 1..r {
        for i in fixed_right[d - 1].min(fixed_right[d])..input.N - 1 {
            cost += chunks[d - 1][i];
            cost += chunks[d][i];
        }
    }
    cost
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
