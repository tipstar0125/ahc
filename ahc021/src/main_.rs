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
    sync::BarrierWaitResult,
    time,
};

use itertools::Itertools;
use num_traits::Pow;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};

const INF: usize = 1 << 60;
const N: usize = 30;
const MAX: usize = 1e4 as usize;

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

#[derive(Debug, Clone)]
struct State {
    B: [[i16; N]; N],
    ascending_locations: Vec<(usize, usize)>,
    turn: usize,
}

impl State {
    fn new(B: [[i16; N]; N]) -> Self {
        let mut ascending_locations = vec![(0, 0); N * (N + 1) / 2];
        for i in 0..N {
            for j in 0..i + 1 {
                ascending_locations[B[i][j] as usize] = (i, j);
            }
        }
        State {
            B,
            ascending_locations,
            turn: 0,
        }
    }
    fn is_done(&self) -> bool {
        self.turn >= MAX
    }
    fn check_and_get_swap_index(&self, pos: (usize, usize)) -> Option<(usize, usize)> {
        let (x, y) = pos;
        if x == 0 {
            return None;
        }
        let mut v = vec![];
        if x != 0 && self.B[x - 1][y] != -1 && self.B[x][y] < self.B[x - 1][y] {
            v.push((self.B[x - 1][y] - self.B[x][y], (x - 1, y)));
        }
        if x != 0 && y != 0 && self.B[x][y] < self.B[x - 1][y - 1] {
            v.push((self.B[x - 1][y - 1] - self.B[x][y], (x - 1, y - 1)));
        }
        v.sort();
        v.reverse();
        if v.is_empty() || v[0].0 < 0 {
            return None;
        } else {
            return Some(v[0].1);
        }
    }
    fn swap(&mut self, pos0: (usize, usize), pos1: (usize, usize)) {
        let (x0, y0) = pos0;
        let (x1, y1) = pos1;
        let tmp = self.B[x0][y0];
        self.B[x0][y0] = self.B[x1][y1];
        self.B[x1][y1] = tmp;
        self.ascending_locations[self.B[x0][y0] as usize] = (x0, y0);
        self.ascending_locations[self.B[x1][y1] as usize] = (x1, y1);
    }
    fn count_error(&self) -> usize {
        let mut cnt = 0;
        for i in 0..N - 1 {
            for j in 0..i + 1 {
                let b = self.B[i][j];
                if b > self.B[i + 1][j] || b > self.B[i + 1][j + 1] {
                    cnt += 1;
                }
            }
        }
        cnt
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    #[fastout]
    fn solve(&mut self) {
        let mut B = [[-1_i16; N]; N];
        for i in 0..N {
            input! {
                b: [i16; i+1]
            }
            B[i][..(i + 1)].copy_from_slice(&b[..(i + 1)]);
        }
        let start = std::time::Instant::now();
        let time_limit = 1.9;
        let time_keeper = TimeKeeper::new(time_limit);

        let mut state = State::new(B);
        let mut ans = vec![];

        while !time_keeper.isTimeOver() && !state.is_done() && state.count_error() != 0 {
            for i in 0..N * (N + 1) / 2 {
                if let Some(pos1) = state.check_and_get_swap_index(state.ascending_locations[i]) {
                    let pos0 = state.ascending_locations[i];
                    state.swap(pos0, pos1);
                    ans.push((pos0, pos1));
                    state.turn += 1;
                    break;
                }
            }
        }

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        eprintln!("Elapsed time: {}sec", elapsed_time);

        println!("{}", ans.len());
        for (pos0, pos1) in ans {
            println!("{} {} {} {}", pos0.0, pos0.1, pos1.0, pos1.1);
        }
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
