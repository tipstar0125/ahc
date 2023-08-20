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
const DIJ5: [(isize, isize); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
const DIJ9: [(isize, isize); 9] = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
];

#[derive(Debug, Clone)]
struct State {
    L: usize,
    N: usize,
    S: usize,
    YX: Vec<(usize, usize)>,
    P: Vec<Vec<isize>>,
    freezed: Vec<Vec<bool>>,
    temp_pattern: Vec<usize>,
    hot: isize,
    cold: isize,
}

impl State {
    fn new(
        L: usize,
        N: usize,
        S: usize,
        YX: Vec<(usize, usize)>,
        pattern_kind: usize,
        delta: isize,
    ) -> Self {
        let cold = max!(0, MAX / 2 - delta);
        let hot = min!(1000, MAX / 2 + delta);

        let pattern_num = 1 << pattern_kind;
        let mut used = vec![false; pattern_num];
        let mut P = vec![vec![MAX / 2; L]; L];
        let mut freezed = vec![vec![false; L]; L];

        for i in 0..N {
            let (y, x) = YX[i];
            let mut hot_mask = 0;
            let mut cold_mask = 0;
            for bit in 0..pattern_kind {
                let (dy, dx) = DIJ9[bit];
                let ny = ((L as isize + y as isize + dy) % L as isize) as usize;
                let nx = ((L as isize + x as isize + dx) % L as isize) as usize;
                if P[ny][nx] == hot {
                    hot_mask += 1 << bit;
                }
                if P[ny][nx] == cold {
                    cold_mask += 1 << bit;
                }
            }
            for j in 0..pattern_num {
                if (j & hot_mask == hot_mask) && (!j & cold_mask == cold_mask) && !used[j] {
                    used[j] = true;
                    for bit in 0..pattern_kind {
                        let (dy, dx) = DIJ9[bit];
                        let ny = ((L as isize + y as isize + dy) % L as isize) as usize;
                        let nx = ((L as isize + x as isize + dx) % L as isize) as usize;
                        let temp = if (j >> bit) % 2 == 0 { cold } else { hot };
                        P[ny][nx] = temp;
                        freezed[ny][nx] = true;
                    }
                    break;
                }
            }
        }

        let mut set = BTreeSet::new();
        let mut temp_pattern = vec![];
        for i in 0..N {
            let (y, x) = YX[i];
            let mut pattern = 0_usize;
            for bit in 0..pattern_kind {
                let (dy, dx) = DIJ9[bit];
                let ny = ((L as isize + y as isize + dy) % L as isize) as usize;
                let nx = ((L as isize + x as isize + dx) % L as isize) as usize;
                if P[ny][nx] == hot {
                    pattern += 1 << bit;
                }
            }
            set.insert(pattern);
            temp_pattern.push(pattern);
        }
        eprintln!("{}", set.len());

        State {
            L,
            N,
            S,
            YX,
            P,
            freezed,
            temp_pattern,
            hot,
            cold,
        }
    }
    fn climbing(&mut self) {
        for _ in 0..1e7 as usize {
            let y = rnd::gen_range(0, self.L);
            let x = rnd::gen_range(0, self.L);
            if self.freezed[y][x] {
                continue;
            }
            let now_cost = self.calc_single_cost(y, x);
            let now = self.P[y][x];
            let d = rnd::gen_range_isize(5);
            self.P[y][x] = max!(0, min!(MAX, self.P[y][x] + d));
            let next_cost = self.calc_single_cost(y, x);
            if now_cost <= next_cost {
                self.P[y][x] = now;
            }
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

        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            rnd::init(1);
        }

        input! {
            A: [usize; N],
            F: [isize; 10000]
        }

        let mut delta_mp = BTreeMap::new();
        delta_mp.insert(1, 10);
        delta_mp.insert(4, 15);
        delta_mp.insert(9, 18);
        delta_mp.insert(16, 24);
        delta_mp.insert(25, 37);
        delta_mp.insert(36, 42);
        delta_mp.insert(49, 46);
        delta_mp.insert(64, 49);
        delta_mp.insert(81, 62);
        delta_mp.insert(100, 76);
        delta_mp.insert(121, 92);
        delta_mp.insert(144, 110);
        delta_mp.insert(169, 129);
        delta_mp.insert(196, 149);
        delta_mp.insert(225, 171);
        delta_mp.insert(256, 200);
        delta_mp.insert(289, 231);
        delta_mp.insert(324, 269);
        delta_mp.insert(361, 298);
        delta_mp.insert(400, 297);
        delta_mp.insert(441, 333);
        delta_mp.insert(484, 371);
        delta_mp.insert(529, 401);
        delta_mp.insert(576, 395);
        delta_mp.insert(625, 437);
        delta_mp.insert(676, 448);
        delta_mp.insert(729, 484);
        delta_mp.insert(784, 496);
        delta_mp.insert(841, 481);
        delta_mp.insert(900, 487);
        let delta = delta_mp[&S];

        let mut n_mp = BTreeMap::new();
        n_mp.insert(1, 1);
        n_mp.insert(4, 1);
        n_mp.insert(9, 2);
        n_mp.insert(16, 3);
        n_mp.insert(25, 3);
        n_mp.insert(36, 6);
        n_mp.insert(49, 11);
        n_mp.insert(64, 13);
        n_mp.insert(81, 13);
        n_mp.insert(100, 13);
        n_mp.insert(121, 13);
        n_mp.insert(144, 13);
        n_mp.insert(169, 13);
        n_mp.insert(196, 13);
        n_mp.insert(225, 13);
        n_mp.insert(256, 13);
        n_mp.insert(289, 13);
        n_mp.insert(324, 13);
        n_mp.insert(361, 13);
        n_mp.insert(400, 12);
        n_mp.insert(441, 12);
        n_mp.insert(484, 12);
        n_mp.insert(529, 14);
        n_mp.insert(576, 14);
        n_mp.insert(625, 14);
        n_mp.insert(676, 14);
        n_mp.insert(729, 14);
        n_mp.insert(784, 14);
        n_mp.insert(841, 13);
        n_mp.insert(900, 13);
        let n = n_mp[&S];

        let mut pattern_kind = 0;
        let mut nn = N;
        while nn > 0 {
            nn /= 2;
            pattern_kind += 1;
        }

        let mut state = State::new(L, N, S, YX, pattern_kind, delta);
        eprintln!("Place cost before: {}", state.calc_cost());
        state.climbing();
        let place_cost = state.calc_cost();
        eprintln!("Place cost after: {}", place_cost);

        for row in &state.P {
            println!("{}", row.iter().join(" "));
        }

        let hot = state.hot;
        let cold = state.cold;

        let mut E = (0..N).collect_vec();
        let mut cnt = 0;
        let mut measure_cost = 0;

        let eval = |measures: &Vec<isize>| -> isize {
            let sum = measures.iter().sum::<isize>();
            let ave = sum / measures.len() as isize;
            if ave < (cold + hot) / 2 {
                cold
            } else {
                hot
            }
        };

        'outer: for i in 0..N {
            let mut measured_pattern = 0;
            for bit in 0..pattern_kind {
                let (dy, dx) = DIJ9[bit];
                let mut measures = vec![];
                for _ in 0..n {
                    if cnt == 10000 {
                        break 'outer;
                    }
                    println!("{} {} {}", i, dy, dx);

                    let (mut py, mut px) = state.YX[A[i]];
                    let L = state.L as isize;
                    py = ((L + py as isize + dy) % L) as usize;
                    px = ((L + px as isize + dx) % L) as usize;
                    let m = max!(0, min!(1000, state.P[py][px] + F[cnt]));
                    measure_cost += 100 * (10 + dy.abs() + dx.abs());

                    // input! {
                    //     m: isize
                    // }

                    measures.push(m);
                    cnt += 1;
                }
                if eval(&measures) == hot {
                    measured_pattern += 1 << bit;
                }
            }

            for out in 0..N {
                if measured_pattern == state.temp_pattern[out] {
                    E[i] = out;
                }
            }
        }

        eprintln!("Measure count: {}", cnt);
        eprintln!("Measure cost: {}", measure_cost);

        println!("-1 -1 -1");
        println!("{}", E.iter().join("\n"));

        let mut score = 1e14;
        let mut wrong_answer = 0;
        for i in 0..N {
            if E[i] != A[i] {
                score *= 0.8;
                wrong_answer += 1;
            }
        }

        score /= (place_cost as f64 + measure_cost as f64 + 1e5).round();
        eprintln!("WA: {}", wrong_answer);
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
