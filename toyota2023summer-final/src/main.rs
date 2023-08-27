#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
    io::Repeat,
    sync::BarrierWaitResult,
};

use itertools::Itertools;
use num_traits::Pow;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use std::str::Lines;

mod rnd {
    static mut S: usize = 0;
    static MAX: usize = 1e9 as usize;

    #[inline]
    pub fn init(seed: usize) {
        unsafe {
            if seed == 0 {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as usize;
                S = t
            } else {
                S = seed;
            }
        }
    }
    #[inline]
    pub fn gen() -> usize {
        unsafe {
            if S == 0 {
                init(0);
            }
            S ^= S << 7;
            S ^= S >> 9;
            S
        }
    }
    #[inline]
    pub fn gen_range(a: usize, b: usize) -> usize {
        gen() % (b - a) + a
    }
    #[inline]
    pub fn gen_bool() -> bool {
        gen() & 1 == 1
    }
    #[inline]
    pub fn gen_range_isize(a: usize) -> isize {
        let mut x = (gen() % a) as isize;
        if gen_bool() {
            x *= -1;
        }
        x
    }
    #[inline]
    pub fn gen_range_neg_wrapping(a: usize) -> usize {
        let mut x = gen() % a;
        if gen_bool() {
            x = x.wrapping_neg();
        }
        x
    }
    #[inline]
    pub fn gen_float() -> f64 {
        ((gen() % MAX) as f64) / MAX as f64
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
            elapsed_time * 1.5 >= self.time_threshold
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
            elapsed_time * 1.5
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}

const ENTRANCE: usize = 4;
const INF: usize = 1 << 60;
const DIJ4: [(usize, usize); 4] = [(!0, 0), (0, !0), (1, 0), (0, 1)];

#[derive(Debug, Clone)]
struct State {
    D: usize,
    N: usize,
    M: usize,
    barrier: Vec<Vec<bool>>,
    container: Vec<Vec<isize>>,
    container_position: Vec<(usize, usize, usize, usize)>,
}

impl State {
    fn new(D: usize, N: usize, barrier: Vec<Vec<bool>>) -> Self {
        let container = vec![vec![-1; D]; D];
        let M = D * D - 1 - N;
        let container_position = vec![];

        Self {
            D,
            N,
            M,
            barrier,
            container,
            container_position,
        }
    }
    fn output(&mut self) -> Vec<(usize, usize)> {
        self.container_position.sort();
        let mut ret = vec![];
        for &(_, _, y, x) in &self.container_position {
            ret.push((y, x));
        }
        ret
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        let start = std::time::Instant::now();

        let mut stdin =
            proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
        macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));
        input! {
            D: usize,
            N: usize,
        }

        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            rnd::init(0);
        }

        let mut barrier = vec![vec![false; D]; D];
        barrier[0][ENTRANCE] = true;
        for _ in 0..N {
            input! {
                rr: usize,
                rc: usize
            }
            barrier[rr][rc] = true;
        }

        let mut state = State::new(D, N, barrier);
        let mut container = vec![vec![-1; D]; D];

        let mut visited = state.barrier.clone();
        let mut dist = vec![vec![INF; D]; D];
        dist[0][ENTRANCE] = 0;
        let mut Q = VecDeque::new();
        Q.push_back((0_usize, ENTRANCE));
        while let Some((r, c)) = Q.pop_front() {
            for &(dr, dc) in &DIJ4 {
                let row = r.wrapping_add(dr);
                let col = c.wrapping_add(dc);
                if row >= D || col >= D || visited[row][col] {
                    continue;
                }
                visited[row][col] = true;
                dist[row][col] = dist[r][c] + 1;
                Q.push_back((row, col));
            }
        }
        let mut target_dist = vec![];
        for i in 0..D {
            for j in 0..D {
                if dist[i][j] == INF || dist[i][j] == 0 {
                    continue;
                }
                target_dist.push(dist[i][j]);
            }
        }
        target_dist.sort();

        for _ in 0..state.M {
            input! {
                t: usize,
            }

            let mut visited = state.barrier.clone();
            let mut dist = vec![vec![INF; D]; D];
            dist[0][ENTRANCE] = 0;
            let mut Q = VecDeque::new();
            Q.push_back((0_usize, ENTRANCE));
            while let Some((r, c)) = Q.pop_front() {
                for &(dr, dc) in &DIJ4 {
                    let row = r.wrapping_add(dr);
                    let col = c.wrapping_add(dc);
                    if row >= D || col >= D || visited[row][col] {
                        continue;
                    }
                    visited[row][col] = true;
                    dist[row][col] = dist[r][c] + 1;
                    Q.push_back((row, col));
                }
            }

            let mut candidates = vec![];

            for i in 0..D {
                for j in 0..D {
                    if dist[i][j] == INF || dist[i][j] == 0 {
                        continue;
                    }
                    let tt = if t >= target_dist.len() {
                        target_dist.len() - 1
                    } else {
                        t
                    };
                    let dd = (dist[i][j] as isize - target_dist[tt] as isize).abs() as usize;
                    candidates.push((dd, i, j, tt));
                }
            }
            candidates.sort();

            for &(_, y, x, td) in &candidates {
                let mut visited = state.barrier.clone();
                visited[y][x] = true;
                let mut Q = VecDeque::new();
                Q.push_back((0_usize, ENTRANCE));
                while let Some((r, c)) = Q.pop_front() {
                    for &(dr, dc) in &DIJ4 {
                        let row = r.wrapping_add(dr);
                        let col = c.wrapping_add(dc);
                        if row >= D || col >= D || visited[row][col] {
                            continue;
                        }
                        visited[row][col] = true;
                        Q.push_back((row, col));
                    }
                }
                let mut ok = true;
                for i in 0..D {
                    for j in 0..D {
                        ok &= visited[i][j];
                    }
                }
                if ok {
                    container[y][x] = t as isize;
                    state.barrier[y][x] = true;
                    target_dist.remove(td);
                    println!("{} {}", y, x);
                    break;
                }
            }
        }

        state.container = container;

        let mut ans = vec![];
        for _ in 0..state.M {
            let mut visited = state.barrier.clone();
            let mut Q = VecDeque::new();
            let mut found = vec![];
            Q.push_back((0_usize, ENTRANCE));
            while let Some((r, c)) = Q.pop_front() {
                for &(dr, dc) in &DIJ4 {
                    let row = r.wrapping_add(dr);
                    let col = c.wrapping_add(dc);
                    if row >= D || col >= D {
                        continue;
                    }
                    if state.container[row][col] != -1 {
                        found.push((state.container[row][col], row, col));
                    }
                    if visited[row][col] {
                        continue;
                    }
                    visited[row][col] = true;
                    Q.push_back((row, col));
                }
            }
            found.sort();
            let pos = (found[0].1, found[0].2);
            state.barrier[pos.0][pos.1] = false;
            state.container[pos.0][pos.1] = -1;
            ans.push(pos);
        }

        for &a in &ans {
            println!("{} {}", a.0, a.1);
        }

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
    }
}

#[macro_export]
macro_rules! max {
    ($x: expr) => ($x);
    ($x: expr, $( $y: expr ),+) => {
        std::cmp::max($x, max!($( $y ),+))
    }
}
#[macro_export]
macro_rules! min {
    ($x: expr) => ($x);
    ($x: expr, $( $y: expr ),+) => {
        std::cmp::min($x, min!($( $y ),+))
    }
}

fn main() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| Solver::default().solve())
        .unwrap()
        .join()
        .unwrap();
}
