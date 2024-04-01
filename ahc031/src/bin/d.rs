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
    let input = read_input();
    for d in 2..input.D {
        let mut sum = 0;
        for n in 0..input.N {
            sum += input.A[d - 1][n].max(input.A[d][n]).max(input.A[d - 1][n]);
        }
        eprintln!("{}", sum);
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
