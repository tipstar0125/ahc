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
};

use itertools::Itertools;
use num_traits::Pow;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};

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

const DIJ4: [(isize, isize); 4] = [(-1, 0), (0, -1), (1, 0), (0, 1)];

#[derive(Debug, Clone)]
struct State {
    L: usize,
    N: usize,
    YX: Vec<(usize, usize)>,
    P: Vec<Vec<isize>>,
}

impl State {
    fn new(L: usize, N: usize, YX: Vec<(usize, usize)>) -> Self {
        let mut P = vec![vec![MAX / 2; L]; L];
        let window = 3;
        let offset = max!(0, (MAX - window * N as isize) / 2);
        for i in 0..N {
            let temp = (MAX - offset * 2) * (i as isize) / (N as isize) + offset;
            let (y, x) = YX[i];
            P[y][x] = temp;
        }

        State { L, N, YX, P }
    }
    fn calc_single_cost(&self, y: usize, x: usize) -> isize {
        let mut cost = 0;
        for (dy, dx) in &DIJ4 {
            let ny = ((self.L as isize + y as isize + dy) % self.L as isize) as usize;
            let nx = ((self.L as isize + x as isize + dx) % self.L as isize) as usize;
            let diff = self.P[y][x] - self.P[ny][nx];
            cost += diff * diff;
        }
        cost
    }
    fn calc_cost(&self) -> isize {
        let mut cost = 0;
        for y in 0..self.L {
            for x in 0..self.L {
                cost += self.calc_single_cost(y, x);
            }
        }
        cost /= 2;
        cost
    }
    fn climbing(&mut self) {
        let mut set = BTreeSet::new();
        for &(y, x) in &self.YX {
            set.insert((y, x));
        }
        for _ in 0..1e7 as usize {
            let y = rnd::gen_range(0, self.L);
            let x = rnd::gen_range(0, self.L);
            if set.contains(&(y, x)) {
                continue;
            }
            let now_cost = self.calc_single_cost(y, x);
            let now = self.P[y][x];
            let d = rnd::gen_range_isize(10);
            self.P[y][x] = max!(0, min!(MAX, self.P[y][x] + d));
            let next_cost = self.calc_single_cost(y, x);
            if now_cost <= next_cost {
                self.P[y][x] = now;
            }
        }
    }
}

const MAX: isize = 1000;
const INF: isize = 1_isize << 60;

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        let start = std::time::Instant::now();
        let mut stdin =
            proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
        macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));
        input! {
            L: usize,
            N: usize,
            S: usize,
            YX: [(usize, usize); N]
        }

        // input! {
        //     A: [usize; N],
        //     F: [isize; 10000]
        // }

        let mut state = State::new(L, N, YX);
        eprintln!("Place cost before: {}", state.calc_cost());
        state.climbing();
        let place_cost = state.calc_cost();
        eprintln!("Place cost after: {}", place_cost);

        for row in &state.P {
            println!("{}", row.iter().join(" "));
        }

        let mut E = (0..N).collect_vec();

        let mut cnt = 0;
        let n = 5;
        let mut fixed = vec![false; N];
        for i in 0..N {
            let mut s = 0_isize;
            for _ in 0..n {
                println!("{} 0 0", i);
                // let (y, x) = state.YX[A[i]];
                // let m = max!(0, min!(1000, state.P[y][x] + F[cnt]));
                input! {
                    m: isize
                }
                s += m;
                cnt += 1;
            }
            s /= n;

            let mut out = 0;
            let mut diff_min = INF;
            for j in 0..N {
                if fixed[j] {
                    continue;
                }
                let (y, x) = state.YX[j];
                let t = state.P[y][x];
                let diff = (s - t).abs();
                if diff < diff_min {
                    diff_min = diff;
                    out = j;
                }
            }
            E[i] = out;
            fixed[out] = true;
        }
        println!("-1 -1 -1");
        println!("{}", E.iter().join("\n"));

        let score = 1e14 as usize / (place_cost as usize + 1e5 as usize);
        eprintln!("Score: {}", score);

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        eprintln!("Elapsed time: {}sec", elapsed_time);
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
