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
    S: usize,
    YX: Vec<(usize, usize)>,
    P: Vec<Vec<isize>>,
    output_temp: isize,
    others_temp: isize,
    between_cells: Vec<(isize, isize, isize, usize, usize)>, // (d, dy, dx, now, next)
}

impl State {
    fn new(L: usize, N: usize, S: usize, YX: Vec<(usize, usize)>) -> Self {
        let output_temp = MAX / 2 - S as isize / 2;
        let others_temp = output_temp + S as isize + 10;
        let mut P = vec![vec![others_temp; L]; L];
        for &(y, x) in &YX {
            P[y][x] = output_temp;
        }

        let mut between_cells = vec![];

        let calc_diff = |a0: isize, a1: isize| -> isize {
            let diff = a1 - a0;
            if diff.abs() <= L as isize / 2 {
                diff
            } else if diff > 0 {
                -(L as isize - diff.abs())
            } else {
                L as isize - diff.abs()
            }
        };

        for i in 0..N {
            for j in 0..N {
                if i == j {
                    continue;
                }
                let (y0, x0) = YX[i];
                let (y1, x1) = YX[j];
                let dy = calc_diff(y0 as isize, y1 as isize);
                let dx = calc_diff(x0 as isize, x1 as isize);
                let d = dy.abs() + dx.abs();
                between_cells.push((d, dy, dx, i, j));
            }
        }
        between_cells.sort();

        State {
            L,
            N,
            S,
            YX,
            P,
            output_temp,
            others_temp,
            between_cells,
        }
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

        input! {
            A: [usize; N],
            F: [isize; 10000]
        }

        let mut state = State::new(L, N, S, YX);
        let place_cost = state.calc_cost();
        eprintln!("Place cost: {}", place_cost);

        for row in &state.P {
            println!("{}", row.iter().join(" "));
        }

        let mut E = (0..N).collect_vec();

        let mut cnt = 0;
        let n = 10;
        let mut candidates = vec![];

        // eprintln!("output temp: {}", state.output_temp);
        // eprintln!("others temp: {}", state.others_temp);

        for i in 0..1 {
            let mut candidate_y = 0;
            let mut candidate_x = 0;
            let mut used = vec![false; state.between_cells.len()];
            let mut candidate: BTreeSet<_> = (0..N).collect();

            for _ in 0..3 {
                let target = rnd::gen_range(0, state.between_cells.len());
                if used[target] {
                    continue;
                }
                used[target] = true;
                let (_, y, x, _, _) = state.between_cells[target];
                let mut s = 0;
                for _ in 0..n {
                    println!("{} {} {}", i, y, x);
                    let (mut py, mut px) = state.YX[A[i]];
                    let L = state.L as isize;
                    py = ((L + py as isize + y) % L) as usize;
                    px = ((L + px as isize + x) % L) as usize;
                    let m = max!(0, min!(1000, state.P[py][px] + F[cnt]));
                    // input! {
                    //     m: isize
                    // }
                    s += m;
                    cnt += 1;
                }
                s /= n;
                if (state.output_temp - s).abs() > (state.others_temp - s).abs() {
                    candidate_y = y;
                    candidate_x = x;
                }
                for &(_, y, x, now, _) in &state.between_cells {
                    if y == candidate_y && x == candidate_x {
                        candidate.remove(&now);
                    }
                }
            }
            candidates.push(candidate);
        }

        for row in &candidates {
            eprintln!("{}  {:?}", row.len(), row);
        }
        eprintln!("cnt: {}", cnt);

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
