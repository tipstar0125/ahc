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
    dist: Vec<(isize, usize, isize, isize)>,
    hot: isize,
    cold: isize,
}

impl State {
    fn new(L: usize, N: usize, S: usize, YX: Vec<(usize, usize)>, delta: isize) -> Self {
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

        let mut cy = L / 2;
        let mut cx = L / 2;
        let mut d = INF;

        for i in 0..L {
            for j in 0..L {
                let mut s = 0;
                for k in 0..N {
                    let (y, x) = YX[k];
                    let dy = calc_diff(i as isize, y as isize).abs();
                    let dx = calc_diff(j as isize, x as isize).abs();
                    s += dy + dx
                }
                if s < d {
                    d = s;
                    cy = i;
                    cx = j;
                }
            }
        }

        let cold = max!(0, MAX / 2 - delta);
        let hot = min!(1000, MAX / 2 + delta);
        let mut P = vec![vec![cold; L]; L];
        P[cy][cx] = hot;

        let mut dist = vec![];
        for i in 0..N {
            let (y0, x0) = YX[i];
            let dy = calc_diff(y0 as isize, cy as isize);
            let dx = calc_diff(x0 as isize, cx as isize);
            let d = dy.abs() + dx.abs();
            dist.push((d, i, dy, dx));
        }
        dist.sort();

        State {
            L,
            N,
            S,
            YX,
            P,
            dist,
            hot,
            cold,
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

        // input! {
        //     A: [usize; N],
        //     F: [isize; 10000]
        // }

        let mut delta_mp = BTreeMap::new();
        delta_mp.insert(1, 10);
        delta_mp.insert(4, 14);
        delta_mp.insert(9, 31);
        delta_mp.insert(16, 56);
        delta_mp.insert(25, 87);
        delta_mp.insert(36, 125);
        delta_mp.insert(49, 171);
        delta_mp.insert(64, 223);
        delta_mp.insert(81, 282);
        delta_mp.insert(100, 348);
        delta_mp.insert(121, 421);
        delta_mp.insert(144, 334);
        delta_mp.insert(169, 393);
        delta_mp.insert(196, 455);
        delta_mp.insert(225, 451);
        delta_mp.insert(256, 479);
        let delta = if delta_mp.contains_key(&S) {
            delta_mp[&S]
        } else {
            3
        };

        let mut n_mp = BTreeMap::new();
        n_mp.insert(1, 1);
        n_mp.insert(4, 1);
        n_mp.insert(9, 1);
        n_mp.insert(16, 1);
        n_mp.insert(25, 1);
        n_mp.insert(36, 1);
        n_mp.insert(49, 1);
        n_mp.insert(64, 1);
        n_mp.insert(81, 1);
        n_mp.insert(100, 1);
        n_mp.insert(121, 1);
        n_mp.insert(144, 2);
        n_mp.insert(169, 2);
        n_mp.insert(196, 2);
        n_mp.insert(225, 3);
        n_mp.insert(256, 3);
        let n = if n_mp.contains_key(&S) { n_mp[&S] } else { 480 };

        let state = State::new(L, N, S, YX, delta);
        let place_cost = state.calc_cost();
        eprintln!("Place cost: {}", place_cost);

        for row in &state.P {
            println!("{}", row.iter().join(" "));
        }

        let mut E = (0..N).collect_vec();

        let mut fixed = vec![false; N];
        let mut cnt = 0;
        let mut measure_cost = 0;

        'outer: for &(_, out, dy, dx) in &state.dist {
            for i in 0..N {
                if fixed[i] {
                    continue;
                }

                let mut measures = vec![];
                for _ in 0..n {
                    if cnt == 10000 {
                        break 'outer;
                    }
                    println!("{} {} {}", i, dy, dx);

                    // let (mut py, mut px) = state.YX[A[i]];
                    // let L = state.L as isize;
                    // py = ((L + py as isize + dy) % L) as usize;
                    // px = ((L + px as isize + dx) % L) as usize;
                    // let m = max!(0, min!(1000, state.P[py][px] + F[cnt]));
                    // measure_cost += 100 * (10 + dy.abs() + dx.abs());

                    input! {
                        m: isize
                    }

                    measures.push(m);
                    cnt += 1;
                }

                measures.sort();
                measures.reverse();
                let sum = measures.iter().sum::<isize>();
                let ave = sum / n as isize;

                if (state.hot - ave).abs() < (state.cold - ave).abs() {
                    fixed[i] = true;
                    E[i] = out;
                    break;
                }
            }
        }

        let mut score = 1e14;
        // for i in 0..N {
        //     if E[i] != A[i] {
        //         score *= 0.8;
        //     }
        // }

        eprintln!("cnt: {}", cnt);
        eprintln!("Measure cost after: {}", measure_cost);

        println!("-1 -1 -1");
        println!("{}", E.iter().join("\n"));

        score /= (place_cost as f64 + measure_cost as f64 + 1e5).round();
        eprintln!("Score: {}", score as usize);

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
